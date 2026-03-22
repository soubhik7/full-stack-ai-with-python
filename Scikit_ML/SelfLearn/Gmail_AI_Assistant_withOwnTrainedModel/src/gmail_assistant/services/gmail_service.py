import base64
from typing import List, Dict, Any, Optional
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from src.gmail_assistant.utils.logger import get_logger

logger = get_logger(__name__)

class GmailService:
    """
    Handles direct interaction with the Gmail API.
    Includes methods for fetching, processing, and sending emails.
    """
    def __init__(self, credentials: Credentials, excluded_subjects: Optional[List[str]] = None):
        """
        Initialize the GmailService with valid credentials.
        """
        self.service = build('gmail', 'v1', credentials=credentials)
        self.excluded_subjects = self._normalize_subject_phrases(excluded_subjects)
        logger.info("Gmail service initialized.")

    def _normalize_subject_phrases(self, phrases: Optional[List[str]]) -> List[str]:
        if not phrases:
            return []
        normalized = []
        for p in phrases:
            if not p:
                continue
            s = str(p).strip()
            if s:
                normalized.append(s)
        return normalized

    def _gmail_quote_phrase(self, phrase: str) -> str:
        s = phrase.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").replace("\r", " ").strip()
        return f'"{s}"'

    def _build_unread_query(self) -> str:
        q = "is:unread"
        for subject in self.excluded_subjects:
            q += f" -subject:{self._gmail_quote_phrase(subject)}"
        return q

    def is_subject_excluded(self, subject: str) -> bool:
        if not self.excluded_subjects:
            return False
        s = (subject or "").strip().lower()
        for excluded in self.excluded_subjects:
            if excluded.strip().lower() in s:
                return True
        return False

    def get_unread_messages(self, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Fetches a list of unread message summaries.
        """
        try:
            q = self._build_unread_query()
            results = self.service.users().messages().list(userId='me', q=q, maxResults=max_results).execute()
            messages = results.get('messages', [])
            logger.info(f"Found {len(messages)} unread messages.")
            return messages
        except Exception as e:
            logger.error(f"Error fetching messages: {e}")
            return []

    def get_message_content(self, msg_id: str) -> Dict[str, Any]:
        """
        Retrieves the full content and metadata for a specific message.
        """
        try:
            message_full = self.service.users().messages().get(userId='me', id=msg_id, format='full').execute()
            payload = message_full.get('payload', {})
            headers = payload.get('headers', [])
            
            # Extract basic metadata
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown Sender')
            
            # Extract body content
            body = self._extract_body(payload)
            if not body:
                body = message_full.get('snippet', '')
            
            return {
                'id': msg_id,
                'subject': subject,
                'sender': sender,
                'body': body
            }
        except Exception as e:
            logger.error(f"Error fetching content for message {msg_id}: {e}")
            raise e

    def _extract_body(self, payload: Dict[str, Any]) -> str:
        """
        Recursively extract the plain text body from a Gmail message payload.
        """
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    data = part['body'].get('data')
                    if data:
                        # Handle base64 padding issues
                        data += "=" * ((4 - len(data) % 4) % 4)
                        return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                elif part['mimeType'] in ['multipart/alternative', 'multipart/related', 'multipart/mixed']:
                    body = self._extract_body(part)
                    if body:
                        return body
        elif payload.get('mimeType') == 'text/plain':
            data = payload.get('body', {}).get('data')
            if data:
                data += "=" * ((4 - len(data) % 4) % 4)
                return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
        return ""

    def mark_as_read(self, msg_id: str):
        """
        Marks a specific message as read by removing the UNREAD label.
        """
        try:
            self.service.users().messages().modify(
                userId='me', 
                id=msg_id, 
                body={'removeLabelIds': ['UNREAD']}
            ).execute()
            logger.info(f"Message {msg_id} marked as read.")
        except Exception as e:
            logger.error(f"Error marking message {msg_id} as read: {e}")

    def send_email(self, to: str, subject: str, body_html: str):
        """
        Sends an email with HTML content.
        """
        from email.mime.text import MIMEText
        from email.utils import formatdate, make_msgid
        
        try:
            msg = MIMEText(body_html, "html", "utf-8")
            msg["To"] = to
            msg["From"] = to # Sending to oneself
            msg["Subject"] = subject
            msg["Date"] = formatdate(localtime=True)
            msg["Message-ID"] = make_msgid(domain="gmail.com")
            
            raw_msg = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
            self.service.users().messages().send(userId='me', body={'raw': raw_msg}).execute()
            logger.info(f"Email sent successfully to {to}.")
        except Exception as e:
            logger.error(f"Error sending email: {e}")

    def get_user_email(self) -> str:
        """
        Retrieves the authenticated user's email address.
        """
        profile = self.service.users().getProfile(userId='me').execute()
        return profile.get('emailAddress', '')
