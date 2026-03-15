import requests
import os
import pickle
import base64
from email.mime.text import MIMEText
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

CLIENT_ID = os.environ["GMAIL_CLIENT_ID"]
CLIENT_SECRET = os.environ["GMAIL_CLIENT_SECRET"]
REFRESH_TOKEN = os.environ["GMAIL_REFRESH_TOKEN"]

print("Loading summarization model...")
# Using Qwen 0.5B Instruct for human-like summaries and entity extraction
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
# Load in float16 for CPU compatibility on GitHub Actions runners to save memory
model_hf = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
# We avoid using pipeline() since the GitHub Actions environment appears to have issues registering it

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

print("Getting user profile...")
profile_url = "https://gmail.googleapis.com/gmail/v1/users/me/profile"
profile_res = requests.get(profile_url, headers=headers).json()
user_email = profile_res.get("emailAddress", "me")

print("\nChecking for delete commands...")
delete_query = f"subject:DELETE_MAIL_ is:unread"
delete_msgs_response = requests.get(f"https://gmail.googleapis.com/gmail/v1/users/me/messages?q={delete_query}", headers=headers).json()
delete_msgs = delete_msgs_response.get("messages", [])

if delete_msgs:
    print(f"Found {len(delete_msgs)} delete commands.")
    for d_msg in delete_msgs:
        d_msg_id = d_msg["id"]
        d_msg_url = f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{d_msg_id}?format=metadata&metadataHeaders=Subject&metadataHeaders=From"
        d_msg_data = requests.get(d_msg_url, headers=headers).json()
        
        subject_header = ""
        from_header = ""
        for header in d_msg_data.get("payload", {}).get("headers", []):
            if header["name"] == "Subject":
                subject_header = header["value"]
            elif header["name"] == "From":
                from_header = header["value"]
                
        # Only process if it really came from us (or our email is in the From header)
        if subject_header.startswith("DELETE_MAIL_") and user_email in from_header:
            target_msg_id = subject_header.replace("DELETE_MAIL_", "").strip()
            
            # Trash the target message
            trash_url = f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{target_msg_id}/trash"
            trash_res = requests.post(trash_url, headers=headers)
            if trash_res.status_code == 200:
                print(f"Successfully trashed message {target_msg_id}.")
            else:
                print(f"Failed to trash message {target_msg_id}.")
                
        # Trash the command email itself
        requests.post(f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{d_msg_id}/trash", headers=headers)

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
    
    # Run the HuggingFace model to generate summary
    print("\nGenerating summary...")
    try:
        # Truncate body if it's too long to avoid exceeding model token limits (distilbart-cnn-12-6 max length is 1024)
        # We'll roughly estimate tokens with character count (3000 chars ~ 750 tokens)
        truncated_body = body[:3000] if len(body) > 3000 else body
        
        # If the email is very short, no need to summarize
        if len(truncated_body.strip()) < 50:
            summary = truncated_body
            print("\nEmail too short to summarize, using original text.")
        else:
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant. Summarize the following email in a brief, professional, third-person narrative. Mention who contacted the user and their main points or requests. Do not include any introductory conversation (e.g., 'Here is the summary:'). Just provide the summary directly."},
                {"role": "user", "content": f"EMAIL SENDER: {sender}\nEMAIL SUBJECT: {subject}\nEMAIL BODY:\n{truncated_body}"}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt")
            
            # Generate response
            generation_output = model_hf.generate(
                **model_inputs,
                max_new_tokens=150,
                do_sample=False, # Use greedy decoding for consistent summaries
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Extract just the newly generated text
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generation_output)
            ]
            summary = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
        print(f"\n--- SUMMARY ---\n{summary}\n---------------")
    except Exception as e:
        print(f"Error summarising email: {e}")
        print(f"Falling back to snippet or original body.")
        summary = body[:200] + "..." if len(body) > 200 else body
    
    # Append to compiled TODOs as dictionary for HTML generation later
    clean_summary = summary.replace(chr(10), ' ')
    compiled_todos.append({
        'id': msg_id,
        'sender': sender,
        'subject': subject,
        'summary': clean_summary
    })
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
    
    # Build a beautiful HTML response instead of a plain table
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Your AI Email Summaries</title>
        <style>
          body {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            background-color: #f4f7f6;
            margin: 0;
            padding: 20px 0;
            color: #333333;
          }
          .container {
            max-width: 600px;
            margin: 0 auto;
            background-color: #ffffff;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
          }
          .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px 20px;
            text-align: center;
            color: #ffffff;
          }
          .header h1 {
            margin: 0;
            font-size: 24px;
            font-weight: 600;
            letter-spacing: 0.5px;
          }
          .header p {
            margin: 10px 0 0;
            font-size: 14px;
            opacity: 0.9;
          }
          .content {
            padding: 20px 25px;
          }
          .email-card {
            background-color: #ffffff;
            border: 1px solid #e1e5eb;
            border-radius: 6px;
            margin-bottom: 20px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            display: flex;
            overflow: hidden;
          }
          .email-card:hover {
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            transform: translateY(-2px);
          }
          .side-column {
            background-color: #667eea;
            width: 36px;
            display: flex;
            align-items: center;
            justify-content: center;
            text-decoration: none;
            color: white;
            font-size: 11px;
            font-weight: bold;
            letter-spacing: 1px;
            transition: background-color 0.2s;
          }
          .side-column:hover {
            background-color: #5a67d8;
          }
          .side-column span {
            writing-mode: vertical-rl;
            transform: rotate(180deg);
            white-space: nowrap;
          }
          .card-content {
            flex: 1;
            padding: 16px;
          }
          .action-column {
            background-color: #f9fafb;
            border-left: 1px solid #e1e5eb;
            display: flex;
            flex-direction: column;
            justify-content: center;
            padding: 0 15px;
          }
          .btn-delete {
            background-color: #fee2e2;
            color: #ef4444;
            text-decoration: none;
            padding: 6px 10px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
            transition: background-color 0.2s;
            text-align: center;
            border: 1px solid #fca5a5;
          }
          .btn-delete:hover {
            background-color: #fca5a5;
            color: #b91c1c;
          }
          .email-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 8px;
            border-bottom: 1px solid #f0f2f5;
            padding-bottom: 8px;
          }
          .email-sender {
            font-size: 13px;
            color: #6b7280;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
          }
          .email-subject {
            font-size: 16px;
            font-weight: bold;
            color: #1f2937;
            margin: 0;
            line-height: 1.4;
          }
          .email-summary {
            font-size: 14px;
            color: #4b5563;
            line-height: 1.6;
            margin-top: 10px;
          }
          .footer {
            background-color: #f9fafb;
            padding: 20px;
            text-align: center;
            font-size: 12px;
            color: #9ca3af;
            border-top: 1px solid #e5e7eb;
          }
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
    
    for todo in compiled_todos:
        html_content += f"""
            <div class="email-card">
              <a href="https://mail.google.com/mail/u/0/#all/{todo['id']}" target="_blank" class="side-column" title="Open Original Email">
                <span>OPEN MAIL</span>
              </a>
              <div class="card-content">
                <div class="email-header">
                  <div>
                    <div class="email-sender">{todo['sender']}</div>
                    <h3 class="email-subject">{todo['subject']}</h3>
                  </div>
                </div>
                <div class="email-summary">
                  {todo['summary']}
                </div>
              </div>
              <div class="action-column">
                <a href="mailto:{user_email}?subject=DELETE_MAIL_{todo['id']}&body=Send%20this%20email%20to%20delete%20the%20original%20message." class="btn-delete" title="Delete Email">DELETE</a>
              </div>
            </div>
        """
        
    html_content += f"""
          </div>
          <div class="footer">
            Generated automatically by your Personal AI Assistant.<br>
            You had {len(compiled_todos)} new items to read.
          </div>
        </div>
      </body>
    </html>
    """
    
    # Set subtype to 'html' to render the styling correctly
    message = MIMEText(html_content, 'html')
    message['to'] = user_email
    
    # A nicer sender name helps prevent spam classification
    message['from'] = f"AI Assistant <{user_email}>"
    message['subject'] = 'Your AI Email Summaries & Brief'
    
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
