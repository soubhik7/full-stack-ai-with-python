import base64
import os
import json
import requests
from email.mime.text import MIMEText
from email.utils import formatdate, make_msgid
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from inference import Summarizer

# Load configuration (mirroring existing project)
# We expect GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, GMAIL_REFRESH_TOKEN for OAuth token refreshing
# Or a GMAIL_TOKEN_JSON secret/file.

def get_email_body(payload):
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

def main():
    print("Loading custom summarization model...")
    summarizer = Summarizer(model_path="model.pth")
    
    # Load credentials
    SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
    creds = None
    
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    elif "GMAIL_TOKEN_JSON" in os.environ:
        token_data = json.loads(os.environ["GMAIL_TOKEN_JSON"])
        creds = Credentials.from_authorized_user_info(token_data, SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            print("No valid credentials found. Please run auth.py first.")
            return
            
    service = build('gmail', 'v1', credentials=creds)
    
    # Fetch unread emails
    print("\nFetching unread emails...")
    results = service.users().messages().list(userId='me', q='is:unread').execute()
    messages = results.get('messages', [])
    
    print(f"Found {len(messages)} unread messages.")
    
    compiled_todos = []
    
    for msg in messages:
        msg_id = msg['id']
        print(f"\nProcessing message {msg_id}...")
        
        message_full = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
        payload = message_full.get('payload', {})
        
        # Extract metadata
        headers = payload.get('headers', [])
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
        sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown Sender')
        
        # Extract body
        body = get_email_body(payload)
        if not body:
            body = message_full.get('snippet', '')
            
        print(f"From: {sender}")
        print(f"Subject: {subject}")
        
        # Run custom summarizer
        print("Generating summary...")
        try:
            summary = summarizer.summarize(body, top_n=2)
            if not summary:
                summary = body[:200] + "..."
            print(f"Summary: {summary}")
        except Exception as e:
            print(f"Error summarising email: {e}")
            summary = body[:200] + "..."
            
        compiled_todos.append({
            'sender': sender,
            'subject': subject,
            'summary': summary.replace('\n', ' ')
        })
        
        # Mark email as read
        service.users().messages().modify(
            userId='me',
            id=msg_id,
            body={'removeLabelIds': ['UNREAD']}
        ).execute()
        print("Marked as read.")
        
    # Send summary email if unread emails were processed
    if compiled_todos:
        print("\nSending compiled summaries...")
        profile = service.users().getProfile(userId='me').execute()
        user_email = profile.get('emailAddress')
        
        # Build HTML content (simplified version of original for brevity)
        html_content = f"<h1>AI Email Summaries</h1><p>You have {len(compiled_todos)} summaries:</p>"
        for item in compiled_todos:
            html_content += f"""
            <div style="border: 1px solid #ccc; padding: 10px; margin-bottom: 10px;">
                <strong>From:</strong> {item['sender']}<br>
                <strong>Subject:</strong> {item['subject']}<br>
                <strong>Summary:</strong> {item['summary']}
            </div>
            """
            
        msg = MIMEText(html_content, "html", "utf-8")
        msg["To"] = user_email
        msg["From"] = user_email
        msg["Subject"] = "Your AI Email Summary Brief"
        msg["Date"] = formatdate(localtime=True)
        msg["Message-ID"] = make_msgid(domain="gmail.com")
        
        raw_msg = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
        service.users().messages().send(userId='me', body={'raw': raw_msg}).execute()
        print(f"Summaries sent to {user_email}.")
        
    print("\nDone processing.")

if __name__ == '__main__':
    main()
