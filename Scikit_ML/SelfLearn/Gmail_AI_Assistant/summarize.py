import base64
import pickle
import os
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")
model = pickle.load(open(model_path, "rb"))

def summarize_email(text):
    return model.predict([text])[0]

# Gmail authentication
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

creds = None
if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
elif "GMAIL_TOKEN_JSON" in os.environ:
    # Option for GitHub action: read from secret
    import json
    token_data = json.loads(os.environ["GMAIL_TOKEN_JSON"])
    creds = Credentials.from_authorized_user_info(token_data, SCOPES)
elif "ACCESS_TOKEN" in os.environ:
    # Older fallback
    creds = Credentials(token=os.environ["ACCESS_TOKEN"])

if not creds or not creds.valid:
    raise Exception("No valid credentials found. Please run auth.py to generate token.json or set GMAIL_TOKEN_JSON.")


service = build("gmail", "v1", credentials=creds)

# Get unread emails
results = service.users().messages().list(
    userId="me",
    q="is:unread",
    maxResults=5
).execute()

messages = results.get("messages", [])

summaries = []

for msg in messages:

    message = service.users().messages().get(
        userId="me",
        id=msg["id"],
        format="full"
    ).execute()

    snippet = message["snippet"]

    summary = summarize_email(snippet)

    summaries.append({
        "email_id": msg["id"],
        "summary": summary
    })

    # mark email as read
    service.users().messages().modify(
        userId="me",
        id=msg["id"],
        body={"removeLabelIds": ["UNREAD"]}
    ).execute()

print(summaries)
