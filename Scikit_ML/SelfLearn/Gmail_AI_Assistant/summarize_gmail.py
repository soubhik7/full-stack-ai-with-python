import requests
import os
import pickle
import base64
from email.mime.text import MIMEText

# Import the custom summarizer (needed for pickle load)
from extractive_summarizer import ExtractiveSummarizer

CLIENT_ID = os.environ["GMAIL_CLIENT_ID"]
CLIENT_SECRET = os.environ["GMAIL_CLIENT_SECRET"]
REFRESH_TOKEN = os.environ["GMAIL_REFRESH_TOKEN"]

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

print("Loading model...")
model = pickle.load(open(model_path, "rb"))

# get access token
token_url = "https://oauth2.googleapis.com/token"

token_data = {
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
    "refresh_token": REFRESH_TOKEN,
    "grant_type": "refresh_token"
}

print("Getting access token...")
token_res = requests.post(token_url, data=token_data).json()
access_token = token_res["access_token"]

headers = {
    "Authorization": f"Bearer {access_token}"
}

def get_email_body(payload):
    if 'parts' in payload:
        for part in payload['parts']:
            if part['mimeType'] == 'text/plain':
                data = part['body'].get('data')
                if data:
                    # Pad the base64 string because Gmail API sometimes sends urlsafe b64 without padding
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

# fetch unread emails
gmail_url = "https://gmail.googleapis.com/gmail/v1/users/me/messages?q=is:unread"

print("\nFetching unread emails...")
messages_response = requests.get(gmail_url, headers=headers).json()
messages = messages_response.get("messages", [])

print(f"Found {len(messages)} unread messages.")

# List to hold the compiled TODOs
compiled_todos = []

for msg in messages:
    msg_id = msg["id"]
    print(f"\n{'='*50}")
    print(f"Processing message {msg_id}...")
    
    # get message details
    msg_url = f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{msg_id}?format=full"
    msg_data = requests.get(msg_url, headers=headers).json()
    
    payload = msg_data.get("payload", {})
    
    # Try to extract the body
    body = get_email_body(payload)
    if not body:
        # Fallback to snippet if body is empty or not parsed correctly
        body = msg_data.get("snippet", "")
    
    # Extract headers
    headers_list = payload.get("headers", [])
    subject = "No Subject"
    sender = "Unknown Sender"
    
    for header in headers_list:
        if header["name"] == "Subject":
            subject = header["value"]
        elif header["name"] == "From":
            sender = header["value"]
            
    print(f"From: {sender}")
    print(f"Subject: {subject}")
    
    # Run the .pkl model to generate summary
    print("\nGenerating summary...")
    try:
        summary = model.predict([body])[0]
        print(f"\n--- SUMMARY ---\n{summary}\n---------------")
    except Exception as e:
        print(f"Error summarising email: {e}")
        print(f"Falling back to snippet: {msg_data.get('snippet', '')}")
        summary = msg_data.get('snippet', '')
    
    # Append to compiled TODOs
    todo_item = f"- [ ] Review: {subject}\n      From: {sender}\n      Summary: {summary.replace(chr(10), ' ')}"
    compiled_todos.append(todo_item)
    print("\n[Added to compiled TODO list]")
    
    # Mark email as read by removing UNREAD label
    print("\nMarking email as read...")
    modify_url = f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{msg_id}/modify"
    modify_data = {
        "removeLabelIds": ["UNREAD"]
    }
    modify_res = requests.post(modify_url, headers=headers, json=modify_data)
    
    if modify_res.status_code == 200:
        print("Successfully marked as read. ✅")
    else:
        print(f"Failed to mark as read: {modify_res.text}")
    print(f"{'='*50}")

# Send the compiled TODO list via email
if compiled_todos:
    print("\nSending compiled TODO list via email...")
    
    # First, get the user's email address
    profile_url = "https://gmail.googleapis.com/gmail/v1/users/me/profile"
    profile_res = requests.get(profile_url, headers=headers).json()
    user_email = profile_res.get("emailAddress", "me")
    
    todos_text = "\n\n".join(compiled_todos)
    email_body = f"Here is your compiled reading list for your latest unread emails:\n\n{todos_text}"
    
    message = MIMEText(email_body)
    message['to'] = user_email
    message['from'] = user_email
    message['subject'] = 'Your AI Email Summaries & TODO List'
    
    # Encode as base64 urlsafe
    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
    
    send_url = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
    send_data = {
        "raw": raw_message
    }
    
    send_res = requests.post(send_url, headers=headers, json=send_data)
    
    if send_res.status_code == 200:
        print(f"Successfully sent compiled TODO list to {user_email}. ✅")
    else:
        print(f"Failed to send email: {send_res.text}")

print("\nDone processing all unread emails.")
