import requests
import os

CLIENT_ID = os.environ["GMAIL_CLIENT_ID"]
CLIENT_SECRET = os.environ["GMAIL_CLIENT_SECRET"]
REFRESH_TOKEN = os.environ["GMAIL_REFRESH_TOKEN"]

# get access token
token_url = "https://oauth2.googleapis.com/token"

token_data = {
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
    "refresh_token": REFRESH_TOKEN,
    "grant_type": "refresh_token"
}

token_res = requests.post(token_url, data=token_data).json()
access_token = token_res["access_token"]

headers = {
    "Authorization": f"Bearer {access_token}"
}

# fetch unread emails
gmail_url = "https://gmail.googleapis.com/gmail/v1/users/me/messages?q=is:unread"

messages = requests.get(gmail_url, headers=headers).json()

print(messages)
