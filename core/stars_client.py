"""
STARS Client — Bilkent Student Information System Scraper
=========================================================
OAuth 1.0 + SMS 2FA authentication, HTML parsing via BeautifulSoup.
Sessions expire after ~1 hour.
"""

import logging
import re
import time
from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("core.stars_client")

BASE = "https://stars.bilkent.edu.tr"


@dataclass
class StarsSession:
    session: requests.Session = field(default_factory=requests.Session)
    authenticated: bool = False
    auth_time: float = 0
    student_id: str = ""
    _oauth_token: str = ""
    _phase: str = "idle"  # idle | awaiting_sms | ready
    _sms_hidden: dict = field(default_factory=dict)
    _verify_url: str = ""  # actual verify endpoint (verifySms or verifyEmail)
    _verify_field: str = "SmsVerifyForm[verifyCode]"  # form field name

    @property
    def expired(self) -> bool:
        return time.time() - self.auth_time > 3500  # ~58 min


@dataclass
class StarsCache:
    """Cached STARS data — persists until next /login."""

    user_info: dict = field(default_factory=dict)
    grades: list = field(default_factory=list)
    attendance: list = field(default_factory=list)
    exams: list = field(default_factory=list)
    letter_grades: list = field(default_factory=list)
    schedule: list = field(default_factory=list)
    fetched_at: float = 0


class StarsClient:
    def __init__(self):
        self._sessions: dict[int, StarsSession] = {}
        self._cache: dict[int, StarsCache] = {}  # user_id → cached data

    # ── Auth helpers ──────────────────────────────────────────────────────

    def _get_session(self, user_id: int) -> StarsSession:
        if user_id not in self._sessions:
            self._sessions[user_id] = StarsSession()
        return self._sessions[user_id]

    def is_authenticated(self, user_id: int) -> bool:
        s = self._sessions.get(user_id)
        if not s:
            return False
        if s.expired:
            s.authenticated = False
            return False
        return s.authenticated

    def is_awaiting_sms(self, user_id: int) -> bool:
        s = self._sessions.get(user_id)
        return s is not None and s._phase == "awaiting_sms"

    # ── Login Flow ────────────────────────────────────────────────────────

    def start_login(self, user_id: int, student_id: str, password: str) -> dict:
        """
        Steps 1-6: Navigate OAuth redirects → POST login → arrive at SMS form.
        Returns {"status": "sms_sent"} or {"status": "error", "message": "..."}.
        """
        ss = StarsSession()
        ss.student_id = student_id
        self._sessions[user_id] = ss
        s = ss.session
        s.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9,tr;q=0.8",
            }
        )

        try:
            # 1-4: Follow redirects to login page
            r = s.get(f"{BASE}/srs/", allow_redirects=True, timeout=15)
            logger.info(f"STARS login step1-4: {r.status_code} url={r.url}")
            for i, resp in enumerate(r.history):
                logger.info(f"  init[{i}]: {resp.status_code} → {resp.headers.get('Location', '?')}")
            logger.info(f"STARS cookies after init: {dict(s.cookies)}")

            # Parse login page for hidden fields (CSRF token etc.)
            login_soup = BeautifulSoup(r.text, "html.parser")
            login_form = login_soup.find("form")
            form_data = {
                "LoginForm[username]": student_id,
                "LoginForm[password]": password,
                "yt0": "",
            }
            if login_form:
                for inp in login_form.find_all("input", {"type": "hidden"}):
                    name = inp.get("name")
                    if name and name not in form_data:
                        form_data[name] = inp.get("value", "")
                logger.info(f"STARS login form fields: {list(form_data.keys())}")

            # 5: POST credentials
            login_url = r.url  # Use actual URL (not hardcoded)
            r = s.post(
                login_url,
                data=form_data,
                allow_redirects=True,
                timeout=15,
            )
            logger.info(f"STARS login POST: {r.status_code} url={r.url}")
            logger.info(f"STARS cookies after login: {dict(s.cookies)}")

            # Check if we landed on verification page (SMS or Email)
            is_verify = "verifySms" in r.url or "verifyEmail" in r.url or "verifyCode" in r.text.lower()
            if is_verify:
                ss._phase = "awaiting_sms"

                # Detect verify type and set URL/field accordingly
                if "verifyEmail" in r.url:
                    ss._verify_url = f"{BASE}/accounts/auth/verifyEmail"
                    ss._verify_field = "EmailVerifyForm[verifyCode]"
                    logger.info("STARS: Email verification phase")
                else:
                    ss._verify_url = f"{BASE}/accounts/auth/verifySms"
                    ss._verify_field = "SmsVerifyForm[verifyCode]"
                    logger.info("STARS: SMS verification phase")

                # Extract oauth_token from redirect chain if present
                for resp in r.history:
                    if "oauth_token" in resp.headers.get("Location", ""):
                        m = re.search(r"oauth_token=([^&]+)", resp.headers["Location"])
                        if m:
                            ss._oauth_token = m.group(1)
                # Also check current URL
                m = re.search(r"oauth_token=([^&]+)", r.url)
                if m:
                    ss._oauth_token = m.group(1)

                # Parse verify page for hidden fields
                verify_soup = BeautifulSoup(r.text, "html.parser")
                verify_form = verify_soup.find("form")
                if verify_form:
                    ss._sms_hidden = {}
                    for inp in verify_form.find_all("input", {"type": "hidden"}):
                        name = inp.get("name")
                        if name:
                            ss._sms_hidden[name] = inp.get("value", "")
                    logger.info(f"STARS verify form hidden fields: {ss._sms_hidden}")

                return {"status": "sms_sent"}

            # If we ended up authenticated (no 2FA?)
            if "/srs" in r.url and "login" not in r.url:
                ss.authenticated = True
                ss.auth_time = time.time()
                ss._phase = "ready"
                return {"status": "ok"}

            # Check for error messages in page
            soup = BeautifulSoup(r.text, "html.parser")
            err = soup.find("div", class_="errorSummary")
            if err:
                return {"status": "error", "message": err.get_text(strip=True)}

            return {"status": "error", "message": "Beklenmeyen sayfa. Giriş başarısız olabilir."}

        except requests.RequestException as e:
            logger.error(f"STARS login error: {e}")
            return {"status": "error", "message": f"Bağlantı hatası: {e}"}

    def verify_sms(self, user_id: int, code: str) -> dict:
        """
        Step 7-9: Submit SMS code → complete OAuth flow → land on /srs/.
        """
        ss = self._sessions.get(user_id)
        if not ss or ss._phase != "awaiting_sms":
            return {"status": "error", "message": "SMS doğrulama beklenmiyordu."}

        s = ss.session

        try:
            # Build verify form data — use stored field name (SMS or Email)
            verify_data = {ss._verify_field: code, "yt0": ""}
            if ss._sms_hidden:
                verify_data.update(ss._sms_hidden)

            # POST code with Referer (browser-like)
            verify_url = ss._verify_url or f"{BASE}/accounts/auth/verifySms"
            s.headers["Referer"] = verify_url
            s.headers["Origin"] = BASE

            r = s.post(verify_url, data=verify_data, allow_redirects=False, timeout=15)
            logger.info(f"STARS verify: {r.status_code} Location={r.headers.get('Location', 'none')}")

            # If 200 with verify page → wrong code
            if r.status_code == 200:
                if "verify" in r.url.lower():
                    return {"status": "error", "message": "Yanlış doğrulama kodu."}

            # Remove stale verification cookie
            try:
                from requests.cookies import remove_cookie_by_name

                remove_cookie_by_name(s.cookies, "verification")
            except (AttributeError, KeyError, ValueError) as exc:
                logger.debug("Verification cookie cleanup skipped: %s", exc)

            # Follow redirect chain manually with Referer headers
            loc = r.headers.get("Location", "")
            if loc.startswith("/"):
                loc = f"{BASE}{loc}"
            logger.info(f"STARS cookies for redirect: {list(s.cookies.keys())}")

            prev_url = verify_url
            max_hops = 15
            r2 = None
            while loc and max_hops > 0:
                s.headers["Referer"] = prev_url
                r2 = s.get(loc, allow_redirects=False, timeout=15)
                logger.info(
                    f"STARS hop: {loc} → {r2.status_code} Location={r2.headers.get('Location', 'none')} Set-Cookie={r2.headers.get('Set-Cookie', 'none')[:100] if r2.headers.get('Set-Cookie') else 'none'}"
                )

                # Remove stale verification cookies set by intermediate redirects
                try:
                    remove_cookie_by_name(s.cookies, "verification")
                except (AttributeError, KeyError, ValueError) as exc:
                    logger.debug("Redirect verification cookie cleanup skipped: %s", exc)

                if r2.status_code in (301, 302, 303, 307):
                    prev_url = loc
                    loc = r2.headers.get("Location", "")
                    if loc.startswith("/"):
                        loc = f"{BASE}{loc}"
                    max_hops -= 1
                else:
                    break

            if r2 is None:
                r2 = r

            logger.info(f"STARS final: {r2.status_code} url={r2.url}")

            # Check if we made it to /srs
            if "/srs" in r2.url and "login" not in r2.url and "accounts" not in r2.url:
                ss.authenticated = True
                ss.auth_time = time.time()
                ss._phase = "ready"
                logger.info(f"STARS authenticated for user {user_id}")
                return {"status": "ok"}

            # If we landed on login page, log the page to understand why
            if r2.status_code == 200 and "login" in r2.url:
                soup = BeautifulSoup(r2.text, "html.parser")
                title = soup.find("title")
                logger.info(f"STARS login page title: {title.get_text() if title else 'none'}")
                # Check if there's a form with specific action
                form = soup.find("form")
                if form:
                    logger.info(f"STARS login page form action: {form.get('action', 'none')}")

            # Last resort: try /srs/ fresh
            s.headers["Referer"] = f"{BASE}/accounts/"
            r3 = s.get(f"{BASE}/srs/", allow_redirects=True, timeout=20)
            logger.info(f"STARS /srs/ fresh: {r3.status_code} url={r3.url}")

            if "/srs" in r3.url and "login" not in r3.url and "accounts" not in r3.url:
                ss.authenticated = True
                ss.auth_time = time.time()
                ss._phase = "ready"
                return {"status": "ok"}

            return {"status": "error", "message": "Doğrulama sonrası yönlendirme başarısız."}

        except (requests.RequestException, ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
            logger.error(
                "STARS SMS verify failed for user=%s: %s",
                user_id,
                exc,
                exc_info=True,
                extra={"user_id": user_id, "verify_url": ss._verify_url},
            )
            return {"status": "error", "message": f"Doğrulama hatası: {exc}"}

    def logout(self, user_id: int):
        ss = self._sessions.pop(user_id, None)
        if ss:
            ss.session.close()

    # ── AJAX Data Methods ─────────────────────────────────────────────────

    def _ajax_post(self, user_id: int, endpoint: str) -> BeautifulSoup | None:
        """POST to a STARS AJAX endpoint, return parsed HTML or None."""
        ss = self._sessions.get(user_id)
        if not ss or not ss.authenticated:
            return None
        if ss.expired:
            ss.authenticated = False
            return None

        url = f"{BASE}/srs/ajax/{endpoint}"
        try:
            r = ss.session.post(
                url,
                data={"rndval": str(int(time.time() * 1000))},
                timeout=15,
            )
            if r.status_code != 200:
                logger.warning(f"STARS AJAX {endpoint}: HTTP {r.status_code}")
                return None
            return BeautifulSoup(r.text, "html.parser")
        except requests.RequestException as e:
            logger.error(f"STARS AJAX {endpoint} error: {e}")
            return None

    def _ajax_get(self, user_id: int, endpoint: str) -> BeautifulSoup | None:
        """GET a STARS AJAX endpoint (SPA navigation pattern), return parsed HTML or None."""
        ss = self._sessions.get(user_id)
        if not ss or not ss.authenticated:
            return None
        if ss.expired:
            ss.authenticated = False
            return None

        url = f"{BASE}/srs/ajax/{endpoint}"
        try:
            r = ss.session.get(url, timeout=15)
            if r.status_code != 200:
                logger.debug("STARS AJAX GET %s: HTTP %s", endpoint, r.status_code)
                return None
            return BeautifulSoup(r.text, "html.parser")
        except requests.RequestException as e:
            logger.error("STARS AJAX GET %s error: %s", endpoint, e)
            return None

    def _discover_prog_string(self, user_id: int) -> str | None:
        """
        Find the progString parameter needed for curriculum.php.

        The SPA JavaScript navigation calls:
          paneSplitter.loadContent("center", "ajax/curriculum.php?progString=DEPT,PROG,1&rndval=...", ...)

        We search for this URL pattern in the home page and the main SRS page.
        """
        ss = self._sessions.get(user_id)
        if not ss or not ss.authenticated:
            return None

        _PATTERN = re.compile(r"curriculum\.php\?progString=([^&\"'<>\s]+)")

        sources = []

        # 1. Try the SRS home page — the shell or home fragment may embed navigation links
        try:
            r = ss.session.get(f"{BASE}/srs/", timeout=15)
            sources.append(r.text)
        except requests.RequestException:
            pass

        # 2. Try home.php — the home content panel loaded by completeLogin
        try:
            r = ss.session.get(f"{BASE}/srs/ajax/home.php", timeout=15)
            sources.append(r.text)
        except requests.RequestException:
            pass

        # 3. Try the setup JS — navigation is configured in setup-dhtml.js
        try:
            r = ss.session.get(f"{BASE}/srs/js/setup-dhtml.js", timeout=15)
            sources.append(r.text)
        except requests.RequestException:
            pass

        for text in sources:
            m = _PATTERN.search(text)
            if m:
                from urllib.parse import unquote
                prog = unquote(m.group(1))
                logger.info("STARS: discovered progString=%s", prog)
                return prog

        return None

    # ── User Info + CGPA ──────────────────────────────────────────────────

    def get_user_info(self, user_id: int) -> dict | None:
        soup = self._ajax_post(user_id, "userInfo.php")
        if not soup:
            return None

        info = {}
        rows = soup.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) >= 2:
                key = cells[0].get_text(strip=True).lower()
                val = cells[1].get_text(strip=True)
                if "cgpa" in key:
                    info["cgpa"] = val
                elif "standing" in key:
                    info["standing"] = val
                elif "class" in key:
                    info["class"] = val
                elif "name" in key:
                    info["name"] = val
                elif "surname" in key:
                    info["surname"] = val
                elif "mobile" in key:
                    info["mobile"] = val
                elif "mail" in key or "email" in key:
                    info["email"] = val

        # Combine name
        if "name" in info and "surname" in info:
            info["full_name"] = f"{info['name']} {info['surname']}"

        return info if info else None

    # ── Assessment Grades ─────────────────────────────────────────────────

    def get_grades(self, user_id: int) -> list[dict] | None:
        soup = self._ajax_post(user_id, "gradeAndAttend/grade.php")
        if not soup:
            return None

        courses = []
        # Look for h4 headers (course names) and their following tables
        headers = soup.find_all("h4")
        for h4 in headers:
            course_text = h4.get_text(strip=True)
            no_grades = "no assessment grades" in course_text.lower()

            course_data = {"course": course_text, "assessments": []}

            if no_grades:
                # Extract course name from "No assessment grades found for COURSE"
                m = re.search(r"for\s+(.+)", course_text, re.IGNORECASE)
                if m:
                    course_data["course"] = m.group(1).strip()
                courses.append(course_data)
                continue

            # Find next table after this h4
            table = h4.find_next("table")
            if table:
                rows = table.find_all("tr")
                for row in rows[1:]:  # skip header
                    cells = row.find_all("td")
                    if len(cells) >= 2:
                        assessment = {
                            "name": cells[0].get_text(strip=True),
                            "grade": cells[1].get_text(strip=True),
                        }
                        if len(cells) >= 3:
                            assessment["weight"] = cells[2].get_text(strip=True)
                        course_data["assessments"].append(assessment)

            courses.append(course_data)

        return courses

    # ── Attendance ────────────────────────────────────────────────────────

    def get_attendance(self, user_id: int) -> list[dict] | None:
        soup = self._ajax_post(user_id, "gradeAndAttend/attend.php")
        if not soup:
            return None

        courses = []
        divs = soup.find_all("div", class_="attendDiv")

        # --- DIAGNOSTIC: log HTML structure to identify parsing issues ---
        logger.debug(
            "Attendance page: %d attendDiv found. h4 texts: %s",
            len(divs),
            [d.find("h4").get_text(strip=True)[:40] if d.find("h4") else "NO_H4"
             for d in divs],
        )
        for d in divs:
            _h4 = d.find("h4")
            _h4_text = _h4.get_text(strip=True)[:35] if _h4 else "?"
            _table_in_div = bool(d.find("table"))
            _table_after_h4 = bool(_h4.find_next("table")) if _h4 else False
            logger.debug(
                "  [%s] table_in_div=%s  table_after_h4=%s",
                _h4_text, _table_in_div, _table_after_h4,
            )
        # --- END DIAGNOSTIC ---

        if not divs:
            # Try alternate: look for h4 + table pairs
            divs = [soup]

        for div in divs:
            h4 = div.find("h4")
            if not h4:
                continue

            course_text = h4.get_text(strip=True)
            # Remove "Attendance Records for " prefix
            course_name = re.sub(r"^Attendance Records?\s+for\s+", "", course_text, flags=re.IGNORECASE).strip()

            records = []
            table = div.find("table")
            if table:
                rows = table.find_all("tr")
                for row in rows[1:]:
                    cells = row.find_all("td")
                    if len(cells) >= 3:
                        attended_text = cells[2].get_text(strip=True)
                        # Parse "1 / 1" or "0 / 1"
                        present = "1" in attended_text.split("/")[0] if "/" in attended_text else True
                        records.append(
                            {
                                "title": cells[0].get_text(strip=True),
                                "date": cells[1].get_text(strip=True),
                                "attended": present,
                                "raw": attended_text,
                            }
                        )

            # Look for ratio
            ratio_text = ""
            ratio_div = div.find(string=re.compile(r"Attendance Ratio", re.IGNORECASE))
            if ratio_div:
                ratio_text = ratio_div.strip()
                m = re.search(r"([\d.]+)%", ratio_text)
                if m:
                    ratio_text = m.group(1) + "%"

            courses.append(
                {
                    "course": course_name,
                    "records": records,
                    "ratio": ratio_text,
                }
            )

        return courses

    # ── Scheduled Exams ───────────────────────────────────────────────────

    def get_exams(self, user_id: int) -> list[dict] | None:
        soup = self._ajax_post(user_id, "exam/index.php")
        if not soup:
            return None

        exams = []
        blocks = soup.find_all("div", class_=re.compile(r"corner"))
        if not blocks:
            # Fallback: look for h2 + examTable pairs
            blocks = soup.find_all("div")

        for block in blocks:
            h2 = block.find("h2")
            h3 = block.find("h3")
            if not h2:
                continue

            exam = {
                "course": h2.get_text(strip=True),
                "exam_name": h3.get_text(strip=True) if h3 else "",
                "date": "",
                "time_block": "",
                "time_remaining": "",
            }

            table = block.find("table", class_="examTable")
            if not table:
                table = block.find("table")
            if table:
                for row in table.find_all("tr"):
                    cells = row.find_all("td")
                    if len(cells) >= 2:
                        label = cells[0].get_text(strip=True).lower()
                        val = cells[1].get_text(strip=True)
                        if "remaining" in label:
                            exam["time_remaining"] = val
                        elif "date" in label:
                            exam["date"] = val
                        elif "starting" in label:
                            exam["start_time"] = val
                        elif "reserved" in label or "block" in label:
                            exam["time_block"] = val

            exams.append(exam)

        return exams

    # ── Weekly Schedule ─────────────────────────────────────────────────

    def get_schedule(self, user_id: int) -> list[dict] | None:
        """Fetch weekly class schedule from STARS srs-v2 endpoint."""
        ss = self._sessions.get(user_id)
        if not ss or not ss.authenticated:
            return None
        if ss.expired:
            ss.authenticated = False
            return None

        url = f"{BASE}/srs-v2/schedule/index/weekly"
        try:
            r = ss.session.get(url, timeout=15)
            if r.status_code != 200:
                logger.warning(f"STARS schedule: HTTP {r.status_code}")
                return None
            soup = BeautifulSoup(r.text, "html.parser")
        except requests.RequestException as e:
            logger.error(f"STARS schedule error: {e}")
            return None

        schedule = []

        # Parse schedule table — look for common table structures
        table = soup.find("table")
        if not table:
            # Try finding schedule in div blocks
            logger.debug(f"STARS schedule: no table found, raw length={len(r.text)}")
            return schedule

        rows = table.find_all("tr")
        # First row might be header with day names
        headers = []
        if rows:
            ths = rows[0].find_all(["th", "td"])
            headers = [th.get_text(strip=True) for th in ths]

        day_names = {
            "Monday": "Pazartesi",
            "Tuesday": "Salı",
            "Wednesday": "Çarşamba",
            "Thursday": "Perşembe",
            "Friday": "Cuma",
            "Saturday": "Cumartesi",
            "Pazartesi": "Pazartesi",
            "Salı": "Salı",
            "Çarşamba": "Çarşamba",
            "Perşembe": "Perşembe",
            "Cuma": "Cuma",
            "Cumartesi": "Cumartesi",
        }

        for row in rows[1:]:
            cells = row.find_all("td")
            if not cells:
                continue

            # Try to extract: time slot from first cell, courses from day columns
            time_slot = cells[0].get_text(strip=True) if cells else ""

            for col_idx, cell in enumerate(cells[1:], 1):
                text = cell.get_text(strip=True)
                if not text or text == "-":
                    continue

                day = headers[col_idx] if col_idx < len(headers) else f"Col{col_idx}"
                day_tr = day_names.get(day, day)

                # Extract course code and room from cell text
                parts = text.split()
                course_code = " ".join(parts[:2]) if len(parts) >= 2 else text
                room = parts[-1] if len(parts) >= 3 else ""

                schedule.append(
                    {
                        "day": day_tr,
                        "time": time_slot,
                        "course": course_code,
                        "room": room,
                        "raw": text,
                    }
                )

        logger.info(f"STARS schedule: {len(schedule)} entries parsed")
        return schedule

    # ── Cache System ──────────────────────────────────────────────────────

    def fetch_all_data(self, user_id: int) -> StarsCache | None:
        """Fetch all STARS data and store in cache. Call after successful login."""
        if not self.is_authenticated(user_id):
            return None

        cache = StarsCache(fetched_at=time.time())
        cache.user_info = self.get_user_info(user_id) or {}
        cache.grades = self.get_grades(user_id) or []
        cache.attendance = self.get_attendance(user_id) or []
        cache.exams = self.get_exams(user_id) or []
        cache.letter_grades = self.get_letter_grades(user_id) or []
        cache.schedule = self.get_schedule(user_id) or []

        self._cache[user_id] = cache
        logger.info(
            f"STARS cache built: {len(cache.exams)} exams, "
            f"{len(cache.grades)} course grades, "
            f"{len(cache.attendance)} attendance records, "
            f"{len(cache.schedule)} schedule entries"
        )
        return cache

    def get_cache(self, user_id: int) -> StarsCache | None:
        return self._cache.get(user_id)

    # ── Letter Grades ─────────────────────────────────────────────────────

    def get_transcript(self, user_id: int) -> list[dict] | None:
        """
        Fetch full degree-audit transcript from /srs/ajax/curriculum.php.

        Confirmed endpoint (from browser network inspection):
          GET /srs/ajax/curriculum.php?progString=DEPT,PROG,1&rndval=<timestamp>

        progString (e.g. "CTISS,CTIS_BS,1") identifies the student's curriculum.
        We first try without it (server may infer from session), then discover it.

        Columns: Code(0) | Name(1) | Status(2) | Grade(3) | Credits(4) | Semester(5)
                 | Course Taken Instead(6)   ← elective slots have empty Code cell

        Returns a flat list of graded courses (skips 'Not graded' and S/U/T/W/P).
        """
        ss = self._sessions.get(user_id)
        if not ss or not ss.authenticated:
            return None
        if ss.expired:
            ss.authenticated = False
            return None

        def _fetch_curriculum(extra_params: dict) -> BeautifulSoup | None:
            params = {"rndval": str(int(time.time() * 1000))}
            params.update(extra_params)
            try:
                r = ss.session.get(
                    f"{BASE}/srs/ajax/curriculum.php",
                    params=params,
                    timeout=15,
                )
                if r.status_code == 200 and "Curriculum" in r.text:
                    return BeautifulSoup(r.text, "html.parser")
                logger.debug("curriculum.php %s → HTTP %s (%d bytes)", extra_params, r.status_code, len(r.text))
            except requests.RequestException as exc:
                logger.warning("curriculum.php error: %s", exc)
            return None

        # Strategy 1: Try without progString — server may infer from session cookie
        soup = _fetch_curriculum({})

        # Strategy 2: Discover progString from navigation sources, then retry
        if soup is None:
            prog = self._discover_prog_string(user_id)
            if prog:
                soup = _fetch_curriculum({"progString": prog})

        if soup is None:
            logger.error("STARS transcript: curriculum.php not reachable")
            return None

        courses = []
        # Columns: Code | Name | Status | Grade | Credits | Semester | Course Taken Instead
        # Elective slots have empty Code — actual course is in cell[6]
        for table in soup.find_all("table"):
            for row in table.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) < 5:
                    continue

                def _cell(i: int) -> str:
                    if i >= len(cells):
                        return ""
                    # replace \xa0 (non-breaking space from &nbsp;) with nothing
                    return cells[i].get_text(separator=" ", strip=True).replace("\xa0", "").strip()

                code_text   = _cell(0)
                name_text   = _cell(1)
                status_text = _cell(2)
                grade_text  = _cell(3)
                cred_text   = _cell(4)
                sem_text    = _cell(5)
                taken_text  = _cell(6)  # "MATH 105 Introduction to Calculus I" for elective rows

                # Elective slot row: code is empty, actual course is in taken_text
                if not code_text and taken_text:
                    # Extract code from the beginning: e.g. "MATH 105 ..." → "MATH 105"
                    m = re.match(r"([A-Z]{2,}\s*\d{3}[A-Z]?)\s+(.*)", taken_text)
                    if m:
                        code_text = m.group(1).strip()
                        name_text = m.group(2).strip()

                # Skip header rows and rows with no course code pattern
                if not code_text or not re.match(r"[A-Z]{2,}", code_text):
                    continue

                # Skip "Not graded" rows
                if not grade_text or grade_text.lower() in ("", "-", "not graded", "—", "not graded"):
                    continue

                try:
                    credits = int(cred_text)
                except ValueError:
                    try:
                        credits = int(float(cred_text))
                    except ValueError:
                        credits = 0

                courses.append({
                    "code": code_text,
                    "name": name_text,
                    "status": status_text,
                    "grade": grade_text,
                    "credits": credits,
                    "semester": sem_text,
                })

        logger.info("STARS transcript: %d graded courses parsed", len(courses))
        return courses

    def get_letter_grades(self, user_id: int) -> list[dict] | None:
        soup = self._ajax_post(user_id, "stats/letter-grade.php")
        if not soup:
            return None

        table = soup.find("table", id="letterGrade")
        if not table:
            table = soup.find("table")
        if not table:
            return []

        semesters = []
        current_semester = None

        for row in table.find_all("tr"):
            h4 = row.find("h4")
            if h4:
                if current_semester:
                    semesters.append(current_semester)
                current_semester = {"semester": h4.get_text(strip=True), "courses": []}
                continue

            if current_semester is None:
                continue

            cells = row.find_all("td")
            if len(cells) >= 3:
                dept = cells[0].get_text(strip=True)
                num = cells[1].get_text(strip=True)
                name = cells[2].get_text(strip=True)
                grade = cells[3].get_text(strip=True) if len(cells) >= 4 else ""
                current_semester["courses"].append(
                    {
                        "code": f"{dept} {num}",
                        "name": name,
                        "grade": grade,
                    }
                )

        if current_semester:
            semesters.append(current_semester)

        return semesters
