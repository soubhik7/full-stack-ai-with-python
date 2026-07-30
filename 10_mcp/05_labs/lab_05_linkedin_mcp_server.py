"""
lab_05_linkedin_mcp_server.py — LinkedIn MCP Server
====================================================
An MCP server that lets an LLM read your LinkedIn profile and publish posts
on your behalf, via LinkedIn's official REST API.

This does NOT log into LinkedIn or store your password — it authenticates
with an OAuth2 access token you generate yourself once via
linkedin_oauth_setup.py in this same folder.

Setup:
    1. python 10_mcp/05_labs/linkedin_oauth_setup.py   (one-time, opens your browser)
    2. Copy the printed LINKEDIN_ACCESS_TOKEN into .env

Tools:
  get_my_profile()                → your name, LinkedIn member ID, email
  create_text_post(text)          → publish a text post to your feed
  get_post(post_urn)              → fetch a post's details
  delete_post(post_urn)           → delete a post you created

Notes:
  - create_text_post/delete_post require the "Share on LinkedIn" product
    (w_member_social scope), which may need LinkedIn app review.
  - get_my_profile only needs "Sign In with LinkedIn using OpenID Connect",
    which is auto-approved for any app.

Run standalone:
    python 10_mcp/05_labs/lab_05_linkedin_mcp_server.py

Inspect interactively:
    mcp dev 10_mcp/05_labs/lab_05_linkedin_mcp_server.py
"""

import json
import os
import urllib.parse

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("linkedin-server")

API_BASE = "https://api.linkedin.com"
LINKEDIN_API_VERSION = "202405"  # LinkedIn-Version header, YYYYMM format


def _access_token() -> str:
    token = os.environ.get("LINKEDIN_ACCESS_TOKEN")
    if not token:
        raise RuntimeError(
            "LINKEDIN_ACCESS_TOKEN not set. Run "
            "10_mcp/05_labs/linkedin_oauth_setup.py first and add the "
            "printed token to .env."
        )
    return token


def _headers(rest_api: bool = False) -> dict:
    headers = {
        "Authorization": f"Bearer {_access_token()}",
        "Content-Type": "application/json",
    }
    if rest_api:
        headers["LinkedIn-Version"] = LINKEDIN_API_VERSION
        headers["X-Restli-Protocol-Version"] = "2.0.0"
    return headers


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_my_profile() -> str:
    """
    Get the authenticated user's LinkedIn profile via the OpenID Connect
    userinfo endpoint.

    Returns:
        JSON with member ID ("sub" — used as the author URN for posting),
        name, locale, and email (if the email scope was granted).
    """
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(f"{API_BASE}/v2/userinfo", headers=_headers())
            resp.raise_for_status()
            data = resp.json()

        return json.dumps({
            "member_id": data.get("sub"),
            "name": data.get("name"),
            "given_name": data.get("given_name"),
            "family_name": data.get("family_name"),
            "email": data.get("email"),
            "locale": data.get("locale"),
            "author_urn": f"urn:li:person:{data.get('sub')}",
        }, indent=2)

    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"LinkedIn API error {e.response.status_code}: {e.response.text}"
        )


@mcp.tool()
def create_text_post(text: str, visibility: str = "PUBLIC") -> str:
    """
    Publish a text post to the authenticated user's LinkedIn feed.

    Requires the w_member_social scope (the "Share on LinkedIn" product).

    Args:
        text: The post body.
        visibility: "PUBLIC" or "CONNECTIONS". Defaults to "PUBLIC".

    Returns:
        JSON with the new post's URN (id).
    """
    try:
        with httpx.Client(timeout=10) as client:
            profile_resp = client.get(f"{API_BASE}/v2/userinfo", headers=_headers())
            profile_resp.raise_for_status()
            author_urn = f"urn:li:person:{profile_resp.json()['sub']}"

            body = {
                "author": author_urn,
                "commentary": text,
                "visibility": visibility,
                "distribution": {
                    "feedDistribution": "MAIN_FEED",
                    "targetEntities": [],
                    "thirdPartyDistributionChannels": [],
                },
                "lifecycleState": "PUBLISHED",
                "isReshareDisabledByAuthor": False,
            }

            resp = client.post(
                f"{API_BASE}/rest/posts", headers=_headers(rest_api=True), json=body
            )
            resp.raise_for_status()
            post_urn = resp.headers.get("x-restli-id", "unknown")

        return json.dumps({"status": "published", "post_urn": post_urn}, indent=2)

    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"LinkedIn API error {e.response.status_code}: {e.response.text}"
        )


@mcp.tool()
def get_post(post_urn: str) -> str:
    """
    Fetch details of a LinkedIn post.

    Args:
        post_urn: The post's URN, e.g. "urn:li:share:1234567890".

    Returns:
        JSON with the post's content and metadata.
    """
    encoded_urn = urllib.parse.quote(post_urn, safe="")
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(
                f"{API_BASE}/rest/posts/{encoded_urn}",
                headers=_headers(rest_api=True),
            )
            resp.raise_for_status()
            return json.dumps(resp.json(), indent=2)

    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"LinkedIn API error {e.response.status_code}: {e.response.text}"
        )


@mcp.tool()
def delete_post(post_urn: str) -> str:
    """
    Delete a LinkedIn post you created.

    Args:
        post_urn: The post's URN, e.g. "urn:li:share:1234567890".

    Returns:
        JSON confirming deletion.
    """
    encoded_urn = urllib.parse.quote(post_urn, safe="")
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.delete(
                f"{API_BASE}/rest/posts/{encoded_urn}", headers=_headers(rest_api=True)
            )
            resp.raise_for_status()

        return json.dumps({"status": "deleted", "post_urn": post_urn}, indent=2)

    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"LinkedIn API error {e.response.status_code}: {e.response.text}"
        )


if __name__ == "__main__":
    print("💼 LinkedIn MCP Server starting...")
    print("Tools: get_my_profile, create_text_post, get_post, delete_post")
    print("Requires LINKEDIN_ACCESS_TOKEN in .env — run linkedin_oauth_setup.py first")
    mcp.run()
