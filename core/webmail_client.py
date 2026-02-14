"""
Webmail Client — Bilkent Roundcube IMAP Integration
====================================================
On-demand connection: each operation opens a fresh IMAP session,
does the work, and disconnects. No persistent connection, no keepalive.
"""

import email
import imaplib
import logging
import re
from contextlib import contextmanager
from email.header import decode_header

logger = logging.getLogger("core.webmail_client")

IMAP_HOST = "mail.bilkent.edu.tr"
IMAP_TIMEOUT = 30  # seconds


class WebmailClient:
    def __init__(self):
        self._email: str = ""
        self._password: str = ""
        self._authenticated: bool = False
        self._last_seen_uids: set[bytes] = set()

    @property
    def authenticated(self) -> bool:
        return self._authenticated

    def login(self, email_addr: str, password: str) -> bool:
        """Validate credentials and seed _last_seen_uids."""
        # Store credentials BEFORE _connect() since it reads self._email/_password
        self._email = email_addr
        self._password = password
        try:
            with self._connect() as imap:
                pass  # connection succeeded = credentials valid
            self._authenticated = True
            # Seed: mark existing AIRS/DAIS as seen so we don't spam on first check
            self._last_seen_uids = self._get_airs_dais_uids()
            logger.info(f"IMAP login OK: {email_addr} ({len(self._last_seen_uids)} existing AIRS/DAIS mails)")
            return True
        except Exception as e:
            logger.error(f"IMAP login failed: {e}")
            self._authenticated = False
            self._email = ""
            self._password = ""
            return False

    @contextmanager
    def _connect(self):
        """Open a fresh IMAP connection, yield it, then close."""
        imap = imaplib.IMAP4_SSL(IMAP_HOST, timeout=IMAP_TIMEOUT)
        imap.login(self._email or "", self._password or "")
        imap.select("INBOX")
        try:
            yield imap
        finally:
            try:
                imap.logout()
            except Exception:
                pass

    def _get_airs_dais_uids(self) -> set[bytes]:
        """Get UIDs of all AIRS/DAIS emails (read + unread)."""
        all_uids: set[bytes] = set()
        try:
            with self._connect() as imap:
                for criteria in ['FROM "airs"', 'FROM "dais"', 'SUBJECT "AIRS"', 'SUBJECT "DAIS"']:
                    try:
                        status, data = imap.search(None, f"({criteria})")
                        if status == "OK" and data[0]:
                            all_uids.update(data[0].split())
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"IMAP seed UIDs error: {e}")
        return all_uids

    def check_new_airs_dais(self) -> list[dict]:
        """Check for NEW AIRS/DAIS emails since last check (background job)."""
        if not self._authenticated:
            return []

        new_mails = []
        seen_in_this_batch: set[bytes] = set()

        try:
            with self._connect() as imap:
                for label, searches in [
                    ("AIRS", ['(UNSEEN FROM "airs")', '(UNSEEN SUBJECT "AIRS")']),
                    ("DAIS", ['(UNSEEN FROM "dais")', '(UNSEEN SUBJECT "DAIS")']),
                ]:
                    for criteria in searches:
                        try:
                            status, data = imap.search(None, criteria)
                            if status != "OK" or not data[0]:
                                continue
                            for uid in data[0].split():
                                if uid in self._last_seen_uids or uid in seen_in_this_batch:
                                    continue
                                mail_data = self._fetch_mail(imap, uid)
                                if mail_data:
                                    mail_data["source"] = label
                                    new_mails.append(mail_data)
                                    seen_in_this_batch.add(uid)
                                    self._last_seen_uids.add(uid)
                        except Exception as e:
                            logger.error(f"IMAP search error: {e}")
        except Exception as e:
            logger.error(f"IMAP connection error (check_new): {e}")

        return new_mails

    def check_all_unread(self) -> list[dict]:
        """Check unread AIRS/DAIS emails (/mail command)."""
        if not self._authenticated:
            return []

        mails = []
        seen_uids: set[bytes] = set()

        try:
            with self._connect() as imap:
                for label, searches in [
                    ("AIRS", ['(UNSEEN FROM "airs")', '(UNSEEN SUBJECT "AIRS")']),
                    ("DAIS", ['(UNSEEN FROM "dais")', '(UNSEEN SUBJECT "DAIS")']),
                ]:
                    for criteria in searches:
                        try:
                            status, data = imap.search(None, criteria)
                            if status != "OK" or not data[0]:
                                continue
                            for uid in data[0].split():
                                if uid in seen_uids:
                                    continue
                                mail_data = self._fetch_mail(imap, uid, body=True)
                                if mail_data:
                                    mail_data["source"] = label
                                    mails.append(mail_data)
                                    seen_uids.add(uid)
                        except Exception as e:
                            logger.error(f"IMAP search error: {e}")
        except Exception as e:
            logger.error(f"IMAP connection error (check_all): {e}")

        return mails

    def get_recent_airs_dais(self, limit: int = 3) -> list[dict]:
        """Fetch most recent AIRS/DAIS mails (read or unread)."""
        if not self._authenticated:
            return []

        mails = []
        seen_uids: set[bytes] = set()

        try:
            with self._connect() as imap:
                for label, searches in [
                    ("AIRS", ['FROM "airs"', 'SUBJECT "AIRS"']),
                    ("DAIS", ['FROM "dais"', 'SUBJECT "DAIS"']),
                ]:
                    for criteria in searches:
                        try:
                            status, data = imap.search(None, f"({criteria})")
                            if status != "OK" or not data[0]:
                                continue
                            # Take last N UIDs (most recent)
                            uids = data[0].split()
                            for uid in reversed(uids):
                                if uid in seen_uids:
                                    continue
                                if len(mails) >= limit:
                                    break
                                mail_data = self._fetch_mail(imap, uid, body=True)
                                if mail_data:
                                    mail_data["source"] = label
                                    mails.append(mail_data)
                                    seen_uids.add(uid)
                        except Exception as e:
                            logger.error(f"IMAP search error: {e}")
                    if len(mails) >= limit:
                        break
        except Exception as e:
            logger.error(f"IMAP connection error (get_recent): {e}")

        return mails[:limit]

    @staticmethod
    def _fetch_mail(imap, uid: bytes, body: bool = True) -> dict | None:
        """Fetch single mail headers + optional body preview."""
        try:
            fetch_parts = "(RFC822)" if body else "(BODY.PEEK[HEADER])"
            status, data = imap.fetch(uid, fetch_parts)
            if status != "OK" or not data or not data[0]:
                return None

            raw = data[0][1] if isinstance(data[0], tuple) else data[0]
            msg = email.message_from_bytes(raw)

            # Decode subject
            subject = ""
            raw_subj = msg.get("Subject", "")
            for part, enc in decode_header(raw_subj):
                if isinstance(part, bytes):
                    subject += part.decode(enc or "utf-8", errors="replace")
                else:
                    subject += part

            # Decode from
            from_addr = ""
            raw_from = msg.get("From", "")
            for part, enc in decode_header(raw_from):
                if isinstance(part, bytes):
                    from_addr += part.decode(enc or "utf-8", errors="replace")
                else:
                    from_addr += part

            date_str = msg.get("Date", "")

            result = {
                "uid": uid,
                "from": from_addr,
                "subject": subject,
                "date": date_str,
            }

            if body:
                result["body_preview"] = WebmailClient._extract_body(msg)

            return result

        except Exception as e:
            logger.error(f"IMAP fetch error for uid {uid}: {e}")
            return None

    @staticmethod
    def _extract_body(msg: email.message.Message, max_len: int = 2000) -> str:
        """Extract plain text body, truncated to max_len chars."""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        return payload.decode(charset, errors="replace")[:max_len].strip()
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                return payload.decode(charset, errors="replace")[:max_len].strip()
        return ""

    def fetch_stars_verification_code(self, max_age_seconds: int = 120) -> str | None:
        """Fetch the latest STARS 2FA verification code from email.

        Searches for emails from starsmsg@bilkent.edu.tr, takes the most recent one,
        and extracts the numeric verification code using regex.
        Returns the code string or None if not found.
        """
        if not self._authenticated:
            return None

        try:
            with self._connect() as imap:
                # Search for STARS verification emails
                status, data = imap.search(None, '(FROM "starsmsg@bilkent.edu.tr")')
                if status != "OK" or not data[0]:
                    logger.info("No STARS verification emails found")
                    return None

                # Take the LAST (most recent) UID
                uids = data[0].split()
                latest_uid = uids[-1]

                # Fetch the email
                status, msg_data = imap.fetch(latest_uid, "(RFC822)")
                if status != "OK" or not msg_data or not msg_data[0]:
                    return None

                raw = msg_data[0][1] if isinstance(msg_data[0], tuple) else msg_data[0]
                msg = email.message_from_bytes(raw)

                # Check age — skip if too old
                from email.utils import parsedate_to_datetime
                from datetime import datetime, timezone
                try:
                    mail_date = parsedate_to_datetime(msg.get("Date", ""))
                    age = (datetime.now(timezone.utc) - mail_date).total_seconds()
                    if age > max_age_seconds:
                        logger.info(f"STARS verification email too old: {age:.0f}s > {max_age_seconds}s")
                        return None
                except Exception:
                    pass  # Can't parse date — try anyway

                # Extract body text
                body = self._extract_body(msg)
                if not body:
                    return None

                # Extract verification code: "Verification Code: 50296"
                match = re.search(r'Verification Code:\s*(\d{4,6})', body)
                if match:
                    code = match.group(1)
                    logger.info(f"STARS verification code extracted: {code}")
                    return code

                logger.info(f"No verification code found in email body: {body[:100]}")
                return None

        except Exception as e:
            logger.error(f"IMAP STARS verification code fetch error: {e}")
            return None

    def noop(self):
        """No-op — kept for backward compatibility, does nothing now."""
        pass

    def logout(self):
        """Mark as unauthenticated."""
        self._authenticated = False
        self._email = ""
        self._password = ""
