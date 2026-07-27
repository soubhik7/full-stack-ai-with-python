# =============================================================================
# LAB 21 (NEW — added to fill an AI-103 syllabus gap, not an original course
# file).
#
# ⚠️⚠️⚠️ PREVIEW / BEST-EFFORT — UNVERIFIED CODE ⚠️⚠️⚠️
# A2A ("Agent2Agent") is an open, cross-vendor protocol (not Azure/Foundry-
# specific) for AGENTS TO DISCOVER AND TALK TO OTHER AGENTS across different
# systems/vendors. EXAM_NOTES.md's memory hook, repeated throughout this
# chapter's notes: **"MCP connects agents -> tools. A2A connects agents ->
# agents."** No A2A code exists anywhere else in this repo - this file is
# this lab's own best-effort illustration, modeled on the publicly documented
# A2A spec shape (an "Agent Card" JSON document describing an agent's
# skills, served at a well-known URL, plus a JSON-RPC-based task/message
# exchange). The exact Python package/class names below are NOT guaranteed
# to match whatever `a2a-sdk`-family package is current when you read this -
# check pypi.org / the A2A protocol's official site for the current
# package name and API before relying on this.
#
# Prerequisites (once you verify the real package): `pip install a2a-sdk`
# (or whatever the current official Python package is) + two agents you
# want talking to each other (Foundry agents from earlier labs, or any
# other agent framework's agents - that cross-framework/cross-vendor
# reach is A2A's whole point).
# =============================================================================

import asyncio

try:
    # ⚠️ Best-effort guessed import - confirm the real package name first.
    from a2a.types import AgentCard, AgentSkill, AgentCapabilities  # type: ignore
    from a2a.client import A2AClient                                # type: ignore
    A2A_SDK_AVAILABLE = True
except ImportError:
    A2A_SDK_AVAILABLE = False


def describe_agent_card_concept():
    """An Agent Card is A2A's discovery mechanism: a JSON document any A2A
    agent publishes (conventionally at `/.well-known/agent.json`) that lists
    its name, description, and SKILLS - what it can be asked to do. Another
    agent (or a client app) fetches this card FIRST, before ever sending a
    real task, to learn whether this remote agent can help and how to
    reach it - conceptually similar to how Lab 8's MCPTool discovers a
    server's tools via `list_tools()`, but for whole AGENTS instead of
    individual functions.
    """
    example_agent_card = {
        "name": "astronomy-agent",
        "description": "Looks up astronomical events and calculates telescope rental costs.",
        "url": "https://example.com/agents/astronomy",
        "skills": [
            {
                "id": "next_visible_event",
                "name": "Next visible event",
                "description": "Find the next visible astronomical event for a location.",
            },
            {
                "id": "calculate_observation_cost",
                "name": "Observation cost calculator",
                "description": "Calculate telescope rental cost for an observation session.",
            },
        ],
    }
    print("Example A2A Agent Card (this is real JSON shape per the public A2A spec):")
    import json
    print(json.dumps(example_agent_card, indent=2))


async def call_remote_agent_via_a2a(remote_agent_url, task_text):
    """Illustrative best-effort sketch of an A2A client call.

    Notice how different this is from every other agent-to-agent pattern
    in this chapter: Lab 18's ConnectedAgentTool only works between agents
    IN THE SAME Foundry project; A2A is meant to reach an agent hosted by
    a COMPLETELY DIFFERENT system/vendor, as long as it speaks the A2A
    protocol and publishes a compatible Agent Card.
    """
    if not A2A_SDK_AVAILABLE:
        raise ImportError(
            "The 'a2a' package (or whichever the current official A2A Python SDK is "
            "named) is not installed / could not be imported - this confirms the "
            "import at the top of this file is illustrative, not proven. Check "
            "pypi.org for the current package name."
        )

    # ⚠️ Best-effort guessed call shape.
    client = A2AClient(base_url=remote_agent_url)
    task = await client.send_task(message=task_text)
    return task.result


def main():
    print("=" * 60)
    print("A2A Protocol concept demo")
    print("=" * 60)
    describe_agent_card_concept()

    print("\nAttempting a (best-effort, likely unverified) live A2A call...")
    try:
        result = asyncio.run(
            call_remote_agent_via_a2a(
                remote_agent_url="https://example.com/agents/astronomy",
                task_text="What's the next visible eclipse from north_america?",
            )
        )
        print(f"Remote agent replied: {result}")
    except Exception as ex:
        print(f"(Expected in this preview lab: {ex})")
        print(
            "\nMCP connects agents -> tools. A2A connects agents -> agents. "
            "Both are open, cross-vendor standards - but unlike MCP (proven "
            "working in Lab 8 of this folder), no A2A implementation is "
            "verified to run in this repo."
        )


if __name__ == "__main__":
    main()
