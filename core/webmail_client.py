"""
Webmail Client — Bilkent Roundcube IMAP Integration
====================================================
Monitors INBOX for AIRS (instructor) and DAIS (department) emails.
"""

import email
import imaplib
import logging
from email.header import decode_header

logger = logging.getLogger("core.webmail_client")

IMAP_HOST = "mail.bilkent.edu.tr"


class WebmailClient:
    def __init__(self):
        self._imap: imaplib.IMAP4_SSL | None = None
        self._authenticated: bool = False
        self._email: str = ""
        self._password: str = ""
        self._last_seen_uids: set[bytes] = set()

    @property
    def authenticated(self) -> bool:
        return self._authenticated

    def login(self, email_addr: str, password: str) -> bool:
        """Connect to IMAP and authenticate."""
        try:
            self._imap = imaplib.IMAP4_SSL(IMAP_HOST)
            self._imap.login(email_addr, password)
            self._imap.select("INBOX")
            self._authenticated = True
            self._email = email_addr
            self._password = password
            # Mark existing AIRS/DAIS mails as already seen
            self._last_seen_uids = self._get_airs_dais_uids()
            logger.info(f"IMAP login OK: {email_addr} ({len(self._last_seen_uids)} existing AIRS/DAIS mails)")
            return True
        except Exception as e:
            logger.error(f"IMAP login failed: {e}")
            self._authenticated = False
            return False

    def _get_airs_dais_uids(self) -> set[bytes]:
        """Get UIDs of all AIRS/DAIS emails (read + unread)."""
        if not self._authenticated:
            return set()
        all_uids: set[bytes] = set()
        for criteria in ['FROM "airs"', 'FROM "dais"', 'SUBJECT "AIRS"', 'SUBJECT "DAIS"']:
            try:
                self._imap.select("INBOX")
                status, data = self._imap.search(None, f"({criteria})")
                if status == "OK" and data[0]:
                    all_uids.update(data[0].split())
            except Exception:
                pass
        return all_uids

    def check_new_airs_dais(self) -> list[dict]:
        """
        Check for NEW AIRS/DAIS emails since last check.
        Returns list of {"uid", "from", "subject", "date", "body_preview", "source": "AIRS"|"DAIS"}.
        """
        if not self._authenticated:
            return []

        try:
            self._imap.select("INBOX")
        except Exception:
            self._reconnect()
            return []

        new_mails = []
        seen_in_this_batch: set[bytes] = set()

        for label, searches in [
            ("AIRS", ['(UNSEEN FROM "airs")', '(UNSEEN SUBJECT "AIRS")']),
            ("DAIS", ['(UNSEEN FROM "dais")', '(UNSEEN SUBJECT "DAIS")']),
        ]:
            for criteria in searches:
                try:
                    status, data = self._imap.search(None, criteria)
                    if status != "OK" or not data[0]:
                        continue
                    for uid in data[0].split():
                        if uid in self._last_seen_uids or uid in seen_in_this_batch:
                            continue
                        mail_data = self._fetch_mail(uid)
                        if mail_data:
                            mail_data["source"] = label
                            new_mails.append(mail_data)
                            seen_in_this_batch.add(uid)
                            self._last_seen_uids.add(uid)
                except Exception as e:
                    logger.error(f"IMAP search error: {e}")

        return new_mails

    def check_all_unread(self) -> list[dict]:
        """Check unread AIRS/DAIS emails only. For /mail command."""
        if not self._authenticated:
            return []

        mails = []
        seen_uids: set[bytes] = set()

        for label, searches in [
            ("AIRS", ['(UNSEEN FROM "airs")', '(UNSEEN SUBJECT "AIRS")']),
            ("DAIS", ['(UNSEEN FROM "dais")', '(UNSEEN SUBJECT "DAIS")']),
        ]:
            for criteria in searches:
                try:
                    self._imap.select("INBOX")
                    status, data = self._imap.search(None, criteria)
                    if status != "OK" or not data[0]:
                        continue
                    for uid in data[0].split():
                        if uid in seen_uids:
                            continue
                        mail_data = self._fetch_mail(uid, body=False)
                        if mail_data:
                            mail_data["source"] = label
                            mails.append(mail_data)
                            seen_uids.add(uid)
                except Exception as e:
                    logger.error(f"IMAP search error: {e}")

        return mails

    def _fetch_mail(self, uid: bytes, body: bool = True) -> dict | None:
        """Fetch single mail headers + optional body preview."""
        try:
            fetch_parts = "(RFC822)" if body else "(BODY.PEEK[HEADER])"
            status, data = self._imap.fetch(uid, fetch_parts)
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

            from_addr = msg.get("From", "")
            date_str = msg.get("Date", "")

            result = {
                "uid": uid,
                "from": from_addr,
                "subject": subject,
                "date": date_str,
            }

            if body:
                result["body_preview"] = self._extract_body(msg)

            return result

        except Exception as e:
            logger.error(f"IMAP fetch error for uid {uid}: {e}")
            return None

    @staticmethod
    def _extract_body(msg: email.message.Message) -> str:
        """Extract first 500 chars of plain text body."""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        return payload.decode(charset, errors="replace")[:500].strip()
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                return payload.decode(charset, errors="replace")[:500].strip()
        return ""

    def noop(self):
        """Send IMAP NOOP to keep connection alive. Auto-reconnect on failure."""
        if not self._authenticated:
            return
        try:
            status, _ = self._imap.noop()
            if status != "OK":
                raise Exception("NOOP failed")
        except Exception:
            self._reconnect()

    def _reconnect(self):
        """Reconnect IMAP using stored credentials."""
        if not self._email or not self._password:
            self._authenticated = False
            return
        logger.info("IMAP reconnecting...")
        try:
            if self._imap:
                try:
                    self._imap.logout()
                except Exception:
                    pass
            self._imap = imaplib.IMAP4_SSL(IMAP_HOST)
            self._imap.login(self._email, self._password)
            self._imap.select("INBOX")
            self._authenticated = True
            logger.info("IMAP reconnected successfully")
        except Exception as e:
            logger.error(f"IMAP reconnect failed: {e}")
            self._authenticated = False

    def logout(self):
        if self._imap:
            try:
                self._imap.logout()
            except Exception:
                pass
        self._authenticated = False
        self._email = ""
