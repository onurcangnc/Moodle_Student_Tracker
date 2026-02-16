"""
Moodle API Client
=================
Handles all communication with the Moodle Web Services API.
Fetches courses, sections, content modules, and downloads files (PDF, DOCX, etc.)
"""

import logging
import mimetypes
import os
import re
from dataclasses import dataclass
from pathlib import Path

import requests

from core import config

logger = logging.getLogger(__name__)


# ─── Data Models ─────────────────────────────────────────────────────────────


@dataclass
class MoodleFile:
    """Represents a downloadable file from Moodle."""

    filename: str
    fileurl: str
    filesize: int
    mimetype: str
    module_name: str  # e.g., "Week 3 - Slides"
    section_name: str  # e.g., "Week 3: Buffer Overflow"
    course_name: str
    course_id: int

    @property
    def is_document(self) -> bool:
        ext = Path(self.filename).suffix.lower()
        return ext in {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".txt", ".md", ".html", ".htm", ".rtf", ".odt"}


@dataclass
class CourseSection:
    """Represents a week/topic section in a course."""

    id: int
    name: str
    summary: str
    section_number: int
    modules: list[dict]


@dataclass
class Course:
    """Represents an enrolled course."""

    id: int
    shortname: str
    fullname: str
    sections: list[CourseSection] = None


@dataclass
class Assignment:
    """Represents a Moodle assignment."""

    id: int
    course_id: int
    course_name: str
    name: str
    description: str
    due_date: int  # Unix timestamp (0 = no deadline)
    cutoff_date: int  # Hard deadline
    submitted: bool
    graded: bool
    grade: str  # "Henüz notlanmadı" or actual grade
    max_grade: str
    time_remaining: str  # Human-readable


# ─── API Client ──────────────────────────────────────────────────────────────


class MoodleClient:
    """Moodle Web Services API client."""

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".doc",
        ".pptx",
        ".ppt",
        ".txt",
        ".md",
        ".html",
        ".htm",
        ".rtf",
    }

    TOKEN_FILE = config.data_dir / ".moodle_token"

    def __init__(self):
        self.base_url = config.moodle_url.rstrip("/")
        self.token = self._resolve_token()
        self.api_url = f"{self.base_url}/webservice/rest/server.php"
        self.session = requests.Session()
        self.user_id: int | None = None
        self.site_info: dict = {}

    def _resolve_token(self) -> str:
        """
        Token resolution order:
        1. MOODLE_TOKEN env var (explicit token)
        2. Saved token from previous auto-login (.moodle_token file)
        3. Auto-fetch using MOODLE_USERNAME + MOODLE_PASSWORD
        """
        # 1. Explicit token in .env
        if config.moodle_token:
            return config.moodle_token

        # 2. Previously saved token
        if self.TOKEN_FILE.exists():
            saved = self.TOKEN_FILE.read_text().strip()
            if saved:
                logger.info("Using saved Moodle token from previous login.")
                return saved

        # 3. Auto-fetch with username/password
        username = os.getenv("MOODLE_USERNAME", "")
        password = os.getenv("MOODLE_PASSWORD", "")

        if username and password:
            token = self._fetch_token(username, password)
            if token:
                return token

        logger.error("No Moodle token available. Set MOODLE_TOKEN or " "MOODLE_USERNAME + MOODLE_PASSWORD in .env")
        return ""

    def _fetch_token(self, username: str, password: str) -> str:
        """
        Auto-fetch token from Moodle's token endpoint.
        POST /login/token.php?username=X&password=Y&service=moodle_mobile_app
        """
        token_url = f"{self.base_url}/login/token.php"
        services = ["moodle_mobile_app", "mod_lti_services"]

        for service in services:
            try:
                logger.info(f"Fetching Moodle token (service: {service})...")
                resp = requests.post(
                    token_url,
                    data={
                        "username": username,
                        "password": password,
                        "service": service,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()

                if "token" in data:
                    token = data["token"]
                    # Save for future use (owner-only permissions)
                    self.TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
                    self.TOKEN_FILE.write_text(token)
                    try:
                        os.chmod(self.TOKEN_FILE, 0o600)
                    except OSError:
                        pass  # Windows doesn't support POSIX permissions
                    logger.info(
                        f"✅ Token obtained and saved to {self.TOKEN_FILE}\n"
                        f"   (You won't need to enter credentials again until token expires)"
                    )
                    return token

                error = data.get("error", "Unknown error")
                errorcode = data.get("errorcode", "")
                logger.warning(f"Token fetch failed ({service}): [{errorcode}] {error}")

                # If invalid login, don't try other services
                if errorcode == "invalidlogin":
                    logger.error(
                        "❌ Invalid credentials. Check MOODLE_USERNAME and MOODLE_PASSWORD.\n"
                        "   Note: Bilkent uses SRS authentication. You may need to:\n"
                        "   1. Create a separate Moodle password via Preferences > Change Password\n"
                        "   2. Or use the Moodle Mobile App method (see SETUP_GUIDE.md)"
                    )
                    break

                if errorcode == "enablewsdescription":
                    logger.error(
                        "❌ Web Services disabled on this Moodle instance.\n"
                        "   Contact moodle@bilkent.edu.tr to request API access."
                    )
                    break

            except requests.RequestException as e:
                logger.error(f"Connection error during token fetch: {e}")

        return ""

    @classmethod
    def clear_saved_token(cls):
        """Delete saved token (e.g., if expired or invalid)."""
        if cls.TOKEN_FILE.exists():
            cls.TOKEN_FILE.unlink()
            logger.info("Saved token cleared.")

    # ─── Low-level API ───────────────────────────────────────────────────

    def _call(self, wsfunction: str, **params) -> dict | list:
        """Execute a Moodle Web Services API call."""
        payload = {
            "wstoken": self.token,
            "wsfunction": wsfunction,
            "moodlewsrestformat": "json",
            **params,
        }
        try:
            resp = self.session.post(self.api_url, data=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, dict) and "exception" in data:
                logger.error(f"Moodle API [{wsfunction}]: {data.get('errorcode')} - {data.get('message')}")
                return {}
            return data

        except requests.RequestException as e:
            logger.error(f"Moodle request failed [{wsfunction}]: {e}")
            return {}

    # ─── Authentication & User ───────────────────────────────────────────

    def connect(self) -> bool:
        """Verify connection and retrieve user info."""
        if not self.token:
            logger.error("No token available. Cannot connect.")
            return False

        info = self._call("core_webservice_get_site_info")
        if not info or "userid" not in info:
            # Token might be expired — try re-fetching
            username = os.getenv("MOODLE_USERNAME", "")
            password = os.getenv("MOODLE_PASSWORD", "")
            if username and password:
                logger.warning("Token may be expired. Attempting re-authentication...")
                self.clear_saved_token()
                new_token = self._fetch_token(username, password)
                if new_token:
                    self.token = new_token
                    info = self._call("core_webservice_get_site_info")
                    if info and "userid" in info:
                        logger.info("Re-authentication successful!")
                    else:
                        logger.error("Re-authentication failed.")
                        return False
                else:
                    return False
            else:
                logger.error("Connection failed. Check MOODLE_URL and MOODLE_TOKEN.")
                return False

        self.user_id = info["userid"]
        self.site_info = info  # Store full site info for profile auto-population
        logger.info(f"Connected to '{info.get('sitename')}' as '{info.get('username')}' " f"(uid: {self.user_id})")
        return True

    def keepalive(self):
        """Prevent session timeout with a lightweight API call."""
        try:
            info = self._call("core_webservice_get_site_info")
            if info and "userid" in info:
                return True
        except (ConnectionError, OSError, RuntimeError, ValueError) as exc:
            logger.debug(f"Moodle keepalive probe failed before reconnect: {exc}")
        # Session dead — reconnect
        logger.warning("Moodle keepalive failed, reconnecting...")
        return self.connect()

    # ─── Courses ─────────────────────────────────────────────────────────

    def get_courses(self) -> list[Course]:
        """Get all enrolled courses."""
        if not self.user_id:
            self.connect()

        raw = self._call("core_enrol_get_users_courses", userid=self.user_id)
        if not isinstance(raw, list):
            return []

        courses = []
        for c in raw:
            courses.append(
                Course(
                    id=c["id"],
                    shortname=c.get("shortname", ""),
                    fullname=c.get("fullname", ""),
                )
            )
        logger.info(f"Found {len(courses)} enrolled courses.")
        return courses

    # ─── Course Content (Sections + Modules) ─────────────────────────────

    def get_course_content(self, course_id: int) -> list[CourseSection]:
        """
        Get full course content: sections (weeks/topics) and their modules.
        Uses core_course_get_contents which returns the richest data.
        """
        raw = self._call("core_course_get_contents", courseid=course_id)
        if not isinstance(raw, list):
            return []

        sections = []
        for s in raw:
            section = CourseSection(
                id=s.get("id", 0),
                name=s.get("name", f"Section {s.get('section', '?')}"),
                summary=self._clean_html(s.get("summary", "")),
                section_number=s.get("section", 0),
                modules=s.get("modules", []),
            )
            sections.append(section)
        return sections

    # ─── File Discovery & Download ───────────────────────────────────────

    def discover_files(self, course: Course) -> list[MoodleFile]:
        """
        Discover all downloadable document files in a course.
        Traverses sections → modules → contents to find files.
        """
        sections = self.get_course_content(course.id)
        course.sections = sections
        files = []

        for section in sections:
            for module in section.modules:
                mod_name = module.get("name", "Unnamed")

                # resource, folder, page, url, label — each has different content structure
                contents = module.get("contents", [])
                for content in contents:
                    if content.get("type") != "file":
                        continue

                    filename = content.get("filename", "")
                    ext = Path(filename).suffix.lower()

                    if ext not in self.SUPPORTED_EXTENSIONS:
                        continue

                    mf = MoodleFile(
                        filename=filename,
                        fileurl=content.get("fileurl", ""),
                        filesize=content.get("filesize", 0),
                        mimetype=content.get("mimetype", mimetypes.guess_type(filename)[0] or ""),
                        module_name=mod_name,
                        section_name=section.name,
                        course_name=course.fullname,
                        course_id=course.id,
                    )
                    files.append(mf)

        logger.info(f"[{course.shortname}] Discovered {len(files)} document files.")
        return files

    def download_file(self, moodle_file: MoodleFile, dest_dir: Path) -> Path | None:
        """
        Download a file from Moodle.
        Moodle requires token appended to the URL for file downloads.
        """
        # Organize: data/downloads/{course_id}/{section}/{filename}
        safe_section = self._safe_name(moodle_file.section_name)
        safe_course = self._safe_name(f"{moodle_file.course_id}_{moodle_file.course_name[:40]}")
        target_dir = dest_dir / safe_course / safe_section
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / moodle_file.filename

        # Skip if already downloaded and same size
        if target_path.exists() and target_path.stat().st_size == moodle_file.filesize:
            logger.debug(f"Skipping (cached): {moodle_file.filename}")
            return target_path

        # Moodle file download: append token
        url = moodle_file.fileurl
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}token={self.token}"

        try:
            resp = self.session.get(url, timeout=120, stream=True)
            resp.raise_for_status()

            with open(target_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Downloaded: {moodle_file.filename} ({moodle_file.filesize} bytes)")
            return target_path

        except requests.RequestException as e:
            logger.error(f"Download failed [{moodle_file.filename}]: {e}")
            return None

    def download_all_course_files(self, course: Course) -> list[tuple[MoodleFile, Path]]:
        """Download all document files from a course. Returns (MoodleFile, local_path) pairs."""
        files = self.discover_files(course)
        results = []

        for mf in files:
            local_path = self.download_file(mf, config.downloads_dir)
            if local_path:
                results.append((mf, local_path))

        return results

    # ─── Course Topics / Section Summaries ───────────────────────────────

    def get_course_topics_text(self, course: Course) -> str:
        """
        Extract a structured text representation of a course's content tree.
        Useful for giving the LLM an overview of what was covered.
        """
        if not course.sections:
            course.sections = self.get_course_content(course.id)

        lines = [f"# {course.fullname}\n"]

        for section in course.sections:
            if not section.name or section.name.lower() == "general":
                continue

            lines.append(f"\n## {section.name}")
            if section.summary:
                lines.append(section.summary)

            for mod in section.modules:
                mod_name = mod.get("name", "")
                mod_type = mod.get("modname", "")
                icon = {
                    "resource": "📄",
                    "assign": "📝",
                    "quiz": "❓",
                    "forum": "💬",
                    "url": "🔗",
                    "page": "📃",
                    "folder": "📁",
                    "label": "",
                }.get(mod_type, "•")

                if icon:
                    lines.append(f"  {icon} {mod_name}")

                # Include page/label content if available
                desc = mod.get("description", "")
                if desc:
                    clean = self._clean_html(desc)
                    if clean:
                        lines.append(f"     {clean[:300]}")

        return "\n".join(lines)

    # ─── URL Module Discovery ─────────────────────────────────────────────

    def discover_url_modules(self, course: Course) -> list[dict]:
        """
        Discover URL-type modules from course contents.
        Returns list of dicts with url, name, description, section info.
        """
        if not course.sections:
            course.sections = self.get_course_content(course.id)

        url_modules = []
        for section in course.sections:
            for module in section.modules:
                if module.get("modname") != "url":
                    continue

                mod_name = module.get("name", "")
                description = self._clean_html(module.get("description", ""))

                # URL is in contents[0].fileurl
                url = ""
                for content in module.get("contents", []):
                    if content.get("type") == "url":
                        url = content.get("fileurl", "")
                        break

                if not url:
                    url = module.get("url", "")

                if url:
                    url_modules.append(
                        {
                            "name": mod_name,
                            "url": url,
                            "description": description,
                            "section_name": section.name,
                            "course_name": course.fullname,
                            "course_id": course.id,
                        }
                    )

        logger.info(f"[{course.shortname}] Discovered {len(url_modules)} URL modules.")
        return url_modules

    # ─── Assignments & Deadlines ─────────────────────────────────────────

    def get_assignments(self) -> list[Assignment]:
        """
        Fetch all assignments from enrolled courses with submission status.
        Uses: mod_assign_get_assignments + mod_assign_get_submission_status
        """
        import time as _time

        if not self.user_id:
            self.connect()

        courses = self.get_courses()
        course_ids = [c.id for c in courses]
        course_map = {c.id: c.fullname for c in courses}

        # Fetch all assignments
        params = {f"courseids[{i}]": cid for i, cid in enumerate(course_ids)}
        raw = self._call("mod_assign_get_assignments", **params)

        if not raw or "courses" not in raw:
            return []

        assignments = []
        now = int(_time.time())

        for course_data in raw.get("courses", []):
            cid = course_data.get("id", 0)
            cname = course_map.get(cid, f"Course {cid}")

            for a in course_data.get("assignments", []):
                assign_id = a.get("id", 0)
                due = a.get("duedate", 0)
                cutoff = a.get("cutoffdate", 0)

                # Calculate time remaining
                if due == 0:
                    time_remaining = "Son tarih yok"
                elif due < now:
                    time_remaining = "⏰ Süresi dolmuş!"
                else:
                    delta = due - now
                    days = delta // 86400
                    hours = (delta % 86400) // 3600
                    if days > 0:
                        time_remaining = f"{days} gün {hours} saat"
                    else:
                        time_remaining = f"{hours} saat"

                # Check submission status
                submitted = False
                graded = False
                grade = "Henüz notlanmadı"
                max_grade = ""

                try:
                    status = self._call(
                        "mod_assign_get_submission_status",
                        assignid=assign_id,
                        userid=self.user_id,
                    )
                    if status:
                        # Check submission
                        last_attempt = status.get("lastattempt", {})
                        submission = last_attempt.get("submission", {})
                        if submission.get("status") == "submitted":
                            submitted = True

                        # Check grade
                        feedback = status.get("feedback", {})
                        if feedback:
                            grade_info = feedback.get("grade", {})
                            if grade_info and grade_info.get("grade") is not None:
                                graded = True
                                grade = str(grade_info.get("grade", ""))
                            gradefordisplay = feedback.get("gradefordisplay", "")
                            if gradefordisplay:
                                grade = gradefordisplay

                except (requests.RequestException, KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
                    logger.debug(
                        "Could not fetch submission status for assignment=%s: %s",
                        assign_id,
                        exc,
                        exc_info=True,
                        extra={"assignment_id": assign_id, "course_id": cid},
                    )

                assignments.append(
                    Assignment(
                        id=assign_id,
                        course_id=cid,
                        course_name=cname,
                        name=a.get("name", "Untitled"),
                        description=self._clean_html(a.get("intro", "")),
                        due_date=due,
                        cutoff_date=cutoff,
                        submitted=submitted,
                        graded=graded,
                        grade=grade,
                        max_grade=max_grade,
                        time_remaining=time_remaining,
                    )
                )

        # Sort by due date (soonest first, no-deadline last)
        assignments.sort(key=lambda a: a.due_date if a.due_date > 0 else float("inf"))

        logger.info(f"Found {len(assignments)} assignments across {len(courses)} courses.")
        return assignments

    def get_upcoming_assignments(self, days: int = 14) -> list[Assignment]:
        """Get assignments due in the next N days that haven't been submitted."""
        import time as _time

        now = int(_time.time())
        cutoff = now + (days * 86400)

        all_assignments = self.get_assignments()
        upcoming = [a for a in all_assignments if not a.submitted and 0 < a.due_date <= cutoff]
        return upcoming

    def get_upcoming_events(self, days: int = 30) -> list[dict]:
        """
        Fetch upcoming calendar events (assignments, quizzes, etc.)
        Uses: core_calendar_get_action_events_by_timesort
        """
        import time as _time

        now = int(_time.time())
        end = now + (days * 86400)

        events = self._call(
            "core_calendar_get_action_events_by_timesort",
            timesortfrom=now,
            timesortto=end,
            limitnum=50,
        )

        if not events or "events" not in events:
            return []

        result = []
        for e in events.get("events", []):
            result.append(
                {
                    "name": e.get("name", ""),
                    "course": e.get("course", {}).get("fullname", ""),
                    "type": e.get("modulename", ""),
                    "due_date": e.get("timesort", 0),
                    "url": e.get("url", ""),
                    "action": e.get("action", {}).get("name", ""),
                }
            )

        return result

    # ─── Utilities ───────────────────────────────────────────────────────

    @staticmethod
    def _clean_html(html: str) -> str:
        """Strip HTML tags, decode entities."""
        if not html:
            return ""
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _safe_name(name: str) -> str:
        """Create filesystem-safe directory name."""
        safe = re.sub(r'[<>:"/\\|?*]', "_", name)
        return safe.strip()[:80]
