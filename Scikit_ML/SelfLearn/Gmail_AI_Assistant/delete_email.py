import requests
import os
import sys

CLIENT_ID = os.environ["GMAIL_CLIENT_ID"]
CLIENT_SECRET = os.environ["GMAIL_CLIENT_SECRET"]
REFRESH_TOKEN = os.environ["GMAIL_REFRESH_TOKEN"]

def delete_email(msg_id):
    print(f"Deleting email {msg_id}...")
    token_url = "https://oauth2.googleapis.com/token"
    token_data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "refresh_token": REFRESH_TOKEN,
        "grant_type": "refresh_token"
    }

    print("Getting access token...")
    token_res = requests.post(token_url, data=token_data).json()
    if "access_token" not in token_res:
         print("Failed to get access token")
         sys.exit(1)
         
    access_token = token_res["access_token"]
    
    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    # Trash the target message
    trash_url = f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{msg_id}/trash"
    trash_res = requests.post(trash_url, headers=headers)
    
    if trash_res.status_code == 200:
        print(f"Successfully trashed message {msg_id}.")
    else:
        print(f"Failed to trash message {msg_id}: {trash_res.text}")
        sys.exit(1)

if len(sys.argv) > 1:
    msg_id = sys.argv[1]
    delete_email(msg_id)
else:
    print("Error: No message ID provided.")
    sys.exit(1)
