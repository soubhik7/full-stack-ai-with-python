"""
linkedin_oauth_setup.py — One-time LinkedIn OAuth2 helper
===========================================================
Completes LinkedIn's 3-legged OAuth2 "Authorization Code" flow entirely in
*your own* browser, on your own machine. This script never sees your LinkedIn
password — LinkedIn's login page handles that — it only receives the short-
lived authorization code LinkedIn redirects back to localhost, then exchanges
it for an access token.

Prerequisites (you do this yourself, at https://www.linkedin.com/developers/apps):
  1. Create a LinkedIn Developer App.
  2. Under "Products", request:
       - "Sign In with LinkedIn using OpenID Connect" (auto-approved — gives
         openid, profile, email scopes)
       - "Share on LinkedIn" (gives w_member_social — required to post;
         may require app review / a verified company page)
  3. Under "Auth", add this exact redirect URL:
       http://localhost:8765/callback
  4. Copy the app's Client ID and Client Secret into this repo's .env:
       LINKEDIN_CLIENT_ID=...
       LINKEDIN_CLIENT_SECRET=...

Run:
    python 10_mcp/05_labs/linkedin_oauth_setup.py

It opens your browser to LinkedIn's consent screen. After you approve, it
prints the access token (valid ~60 days) — copy it into .env as
LINKEDIN_ACCESS_TOKEN. Nothing is written to disk automatically.
"""

import os
import secrets
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.environ.get("LINKEDIN_CLIENT_ID")
CLIENT_SECRET = os.environ.get("LINKEDIN_CLIENT_SECRET")
REDIRECT_URI = os.environ.get("LINKEDIN_REDIRECT_URI", "http://localhost:8765/callback")
SCOPES = "openid profile email w_member_social"

AUTH_URL = "https://www.linkedin.com/oauth/v2/authorization"
TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"

_result = {}


class _CallbackHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # silence default request logging

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()

        if "error" in params:
            _result["error"] = params.get("error_description", params["error"])[0]
            self.wfile.write(b"<h2>LinkedIn auth failed. You can close this tab.</h2>")
        else:
            _result["code"] = params.get("code", [None])[0]
            _result["state"] = params.get("state", [None])[0]
            self.wfile.write(b"<h2>LinkedIn auth complete. You can close this tab.</h2>")


def main() -> None:
    if not CLIENT_ID or not CLIENT_SECRET:
        raise SystemExit(
            "Missing LINKEDIN_CLIENT_ID / LINKEDIN_CLIENT_SECRET in .env.\n"
            "Create a LinkedIn Developer App first — see the docstring at the "
            "top of this file."
        )

    state = secrets.token_urlsafe(16)
    query = urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "scope": SCOPES,
            "state": state,
        }
    )
    auth_url = f"{AUTH_URL}?{query}"

    server = HTTPServer(("localhost", 8765), _CallbackHandler)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()

    print("Opening your browser to LinkedIn for login + consent...")
    print(f"If it doesn't open automatically, visit:\n  {auth_url}\n")
    webbrowser.open(auth_url)

    thread.join(timeout=180)

    if "error" in _result:
        raise SystemExit(f"LinkedIn returned an error: {_result['error']}")
    if not _result.get("code"):
        raise SystemExit("Timed out waiting for LinkedIn callback. Try again.")
    if _result.get("state") != state:
        raise SystemExit("State mismatch — possible CSRF, aborting.")

    resp = httpx.post(
        TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": _result["code"],
            "redirect_uri": REDIRECT_URI,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=15,
    )
    resp.raise_for_status()
    payload = resp.json()

    print("\nSuccess! Add this to your .env:\n")
    print(f"LINKEDIN_ACCESS_TOKEN={payload['access_token']}")
    print(f"\n(expires in {payload.get('expires_in', '?')} seconds)")


if __name__ == "__main__":
    main()
