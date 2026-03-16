import base64
import os
import json
import logging
from email.mime.text import MIMEText
from email.utils import formatdate, make_msgid
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

from inference import Summarizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_email_body(payload):
    """Recursively extract the plain text body from a Gmail message payload."""
    if 'parts' in payload:
        for part in payload['parts']:
            if part['mimeType'] == 'text/plain':
                data = part['body'].get('data')
                if data:
                    data += "=" * ((4 - len(data) % 4) % 4)
                    return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
            elif part['mimeType'] in ['multipart/alternative', 'multipart/related', 'multipart/mixed']:
                body = get_email_body(part)
                if body:
                    return body
    elif payload.get('mimeType') == 'text/plain':
        data = payload.get('body', {}).get('data')
        if data:
            data += "=" * ((4 - len(data) % 4) % 4)
            return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
    return ""

def generate_html_brief(compiled_todos):
    """Generates a professional HTML email brief from compiled summaries."""
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Your AI Email Brief</title>
        <style>
          body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; background-color: #f4f7f6; margin: 0; padding: 20px 0; color: #333333; }
          .container { max-width: 600px; margin: 0 auto; background-color: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }
          .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px 20px; text-align: center; color: #ffffff; }
          .header h1 { margin: 0; font-size: 24px; font-weight: 600; letter-spacing: 0.5px; }
          .content { padding: 20px 25px; }
          .email-card { background-color: #ffffff; border: 1px solid #e1e5eb; border-left: 4px solid #667eea; border-radius: 6px; padding: 16px; margin-bottom: 20px; }
          .email-sender { font-size: 13px; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
          .email-subject { font-size: 16px; font-weight: bold; color: #1f2937; margin: 5px 0; line-height: 1.4; }
          .email-summary { font-size: 14px; color: #4b5563; line-height: 1.6; margin-top: 10px; }
          .footer { background-color: #f9fafb; padding: 20px; text-align: center; font-size: 12px; color: #9ca3af; border-top: 1px solid #e5e7eb; }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>📥 AI Email Brief</h1>
            <p>Your latest unread emails, condensed into a quick reading list.</p>
          </div>
          <div class="content">
    """
    
    for item in compiled_todos:
        html_template += f"""
            <div class="email-card">
              <div class="email-sender">{item['sender']}</div>
              <h3 class="email-subject">{item['subject']}</h3>
              <div class="email-summary">{item['summary']}</div>
            </div>
        """
        
    html_template += f"""
          </div>
          <div class="footer">
            Generated automatically by your Personal AI Assistant.<br>
            Processed {len(compiled_todos)} new items.
          </div>
        </div>
      </body>
    </html>
    """
    return html_template

def process_emails(max_emails=5):
    """Fetch unread emails, summarize them, and send a summary brief."""
    logger.info("Starting email processing...")
    
    # Load custom model
    summarizer = Summarizer(model_path="model.pth")
    
    # Authenticate
    SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    elif "GMAIL_TOKEN_JSON" in os.environ:
        token_json = os.environ.get("GMAIL_TOKEN_JSON", "").strip()
        if token_json:
            try:
                token_data = json.loads(token_json)
                creds = Credentials.from_authorized_user_info(token_data, SCOPES)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode GMAIL_TOKEN_JSON: {e}")
    
    # If GMAIL_TOKEN_JSON is not available or failed, check for individual secrets
    if not creds:
        client_id = os.environ.get("GMAIL_CLIENT_ID")
        client_secret = os.environ.get("GMAIL_CLIENT_SECRET")
        refresh_token = os.environ.get("GMAIL_REFRESH_TOKEN")
        
        if client_id and client_secret and refresh_token:
            logger.info("Building credentials from individual GMAIL secrets.")
            token_data = {
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "token_uri": "https://oauth2.googleapis.com/token",
            }
            creds = Credentials.from_authorized_user_info(token_data, SCOPES)
        else:
            missing = []
            if not client_id: missing.append("GMAIL_CLIENT_ID")
            if not client_secret: missing.append("GMAIL_CLIENT_SECRET")
            if not refresh_token: missing.append("GMAIL_REFRESH_TOKEN")
            if missing and "GMAIL_TOKEN_JSON" not in os.environ:
                logger.error(f"Missing required credentials: {', '.join(missing)}")
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            logger.error("No valid credentials found. Run auth.py locally or set GMAIL_TOKEN_JSON.")
            return []
            
    service = build('gmail', 'v1', credentials=creds)
    
    # Fetch unread emails
    results = service.users().messages().list(userId='me', q='is:unread', maxResults=max_emails).execute()
    messages = results.get('messages', [])
    
    if not messages:
        logger.info("No unread emails found.")
        return []
        
    logger.info(f"Found {len(messages)} unread messages.")
    summaries = []
    
    for msg in messages:
        msg_id = msg['id']
        try:
            logger.info(f"Processing message {msg_id}...")
            message_full = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
            payload = message_full.get('payload', {})
            
            # Metadata
            headers = payload.get('headers', [])
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown Sender')
            
            # Content
            body = get_email_body(payload)
            if not body:
                body = message_full.get('snippet', '')
                
            # Summarize
            summary = summarizer.summarize(body, top_n=2)
            if not summary:
                summary = body[:200] + "..."
                
            summaries.append({
                'id': msg_id,
                'sender': sender,
                'subject': subject,
                'summary': summary.replace('\n', ' ')
            })
            
            # Mark as read
            service.users().messages().modify(userId='me', id=msg_id, body={'removeLabelIds': ['UNREAD']}).execute()
            logger.info(f"Successfully processed {msg_id}.")
            
        except Exception as e:
            logger.error(f"Error processing message {msg_id}: {e}")
            continue
            
    if summaries:
        # Send brief
        profile = service.users().getProfile(userId='me').execute()
        user_email = profile.get('emailAddress')
        
        html_content = generate_html_brief(summaries)
        msg = MIMEText(html_content, "html", "utf-8")
        msg["To"] = user_email
        msg["From"] = user_email
        msg["Subject"] = "Your AI Email Summary Brief"
        msg["Date"] = formatdate(localtime=True)
        msg["Message-ID"] = make_msgid(domain="gmail.com")
        
        raw_msg = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
        service.users().messages().send(userId='me', body={'raw': raw_msg}).execute()
        logger.info(f"Summary brief sent to {user_email}.")
        
    return summaries

if __name__ == '__main__':
    process_emails()