"""
App 17 — Multi-Agent Systems
==============================
Three foundational multi-agent patterns built with plain OpenAI calls.
No framework required — understand the patterns before using a framework.

Patterns:
  1. Orchestrator + Workers   — one agent delegates to specialists
  2. Sequential Pipeline      — A → B → C, each builds on the previous
  3. Debate/Critic Loop       — Writer ↔ Critic until quality passes

Run: python 08_ai_apps/17_multi_agent/main.py
Requires: OPENAI_API_KEY in .env
"""

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()
client = OpenAI()


# ── Base Agent class ──────────────────────────────────────────────────────────

class Agent:
    """
    A simple, reusable agent with a fixed persona.

    Args:
        name: Human-readable agent name.
        system_prompt: The system prompt defining this agent's role and behaviour.
        model: OpenAI model to use.
    """

    def __init__(self, name: str, system_prompt: str, model: str = "gpt-4o-mini"):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model

    def run(self, user_message: str, max_tokens: int = 600) -> str:
        """Run the agent on a single user message."""
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    def run_with_history(self, messages: list[dict], max_tokens: int = 600) -> str:
        """Run with a full conversation history (includes system prompt automatically)."""
        full_messages = [{"role": "system", "content": self.system_prompt}] + messages
        response = client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    def __repr__(self) -> str:
        return f"Agent({self.name})"


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 1: ORCHESTRATOR + WORKERS
# One agent receives the task, breaks it into sub-tasks, delegates, then
# synthesises the results into a final answer.
# ══════════════════════════════════════════════════════════════════════════════

print("="*65)
print("PATTERN 1: ORCHESTRATOR + WORKERS")
print("  Orchestrator → Research + Analysis + Writing → Final Report")
print("="*65)

# Define specialised workers
researcher = Agent(
    name="Researcher",
    system_prompt=(
        "You are a research specialist. When given a topic, you gather key facts, "
        "statistics, and background information. Be factual and comprehensive. "
        "Present findings as a structured bullet-point list."
    ),
)

analyst = Agent(
    name="Analyst",
    system_prompt=(
        "You are a business analyst. When given research findings, you identify "
        "trends, implications, and opportunities. Focus on actionable insights. "
        "Present as numbered key insights."
    ),
)

writer = Agent(
    name="Writer",
    system_prompt=(
        "You are a technical writer creating executive-level content. "
        "Transform research and analysis into clear, professional narratives. "
        "Use plain language. Maximum 200 words. Start with a compelling opening sentence."
    ),
)

orchestrator = Agent(
    name="Orchestrator",
    system_prompt=(
        "You are a project manager coordinating a content creation team. "
        "Given a final report compiled from research, analysis, and writing, "
        "you review it and produce a concise executive summary (3-4 sentences) "
        "plus 3 recommended next steps."
    ),
)


def orchestrate(topic: str) -> dict:
    """Run the orchestrator-worker pattern on a topic."""
    print(f"\n📋 Topic: {topic}\n")

    # Step 1: Researcher gathers facts
    print("  🔬 Researcher: gathering facts...")
    research = researcher.run(f"Research: {topic}")

    # Step 2: Analyst interprets findings
    print("  📊 Analyst: identifying insights...")
    analysis = analyst.run(
        f"Topic: {topic}\n\nResearch findings:\n{research}\n\nProvide your analysis."
    )

    # Step 3: Writer creates narrative
    print("  ✍️  Writer: drafting narrative...")
    narrative = writer.run(
        f"Topic: {topic}\n\nFacts:\n{research}\n\nInsights:\n{analysis}\n\nWrite the report."
    )

    # Step 4: Orchestrator synthesises
    print("  🎯 Orchestrator: synthesising final output...")
    final = orchestrator.run(
        f"Topic: {topic}\n\nNarrative:\n{narrative}\n\nProvide executive summary + next steps."
    )

    return {
        "topic": topic,
        "research": research,
        "analysis": analysis,
        "narrative": narrative,
        "executive_summary": final,
    }


result = orchestrate("The impact of Model Context Protocol (MCP) on AI application development")
print(f"\n📝 Research ({len(result['research'])} chars): {result['research'][:100]}...")
print(f"📊 Analysis ({len(result['analysis'])} chars): {result['analysis'][:100]}...")
print(f"✍️  Narrative ({len(result['narrative'])} chars): {result['narrative'][:100]}...")
print(f"\n🎯 EXECUTIVE SUMMARY:\n{result['executive_summary']}")


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 2: SEQUENTIAL PIPELINE
# Each agent transforms the output of the previous agent.
# Useful for: translate → summarise → classify → format
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*65)
print("PATTERN 2: SEQUENTIAL PIPELINE")
print("  Raw text → Cleaner → Summariser → Categoriser → Formatter")
print("="*65)

# Pipeline agents
cleaner = Agent(
    name="Cleaner",
    system_prompt=(
        "You clean and normalise text: fix typos, remove filler words, "
        "fix grammar, standardise formatting. Output only the cleaned text."
    ),
)

summariser = Agent(
    name="Summariser",
    system_prompt=(
        "You summarise text into 3 key points. Each point is one sentence. "
        "Output only the 3 bullet points, nothing else."
    ),
)

categoriser = Agent(
    name="Categoriser",
    system_prompt=(
        "You categorise content into one of: Technology, Business, Science, "
        "Health, Education, Entertainment, Finance. "
        "Also assign 3 relevant tags. "
        "Output format: CATEGORY: X | TAGS: tag1, tag2, tag3"
    ),
)

formatter = Agent(
    name="Formatter",
    system_prompt=(
        "You format content into clean markdown. "
        "Given summary points and category info, create a structured card with: "
        "## Title (infer from content), **Category**, **Tags**, and the bullet points."
    ),
)


def run_pipeline(raw_text: str) -> str:
    """Run text through the sequential pipeline."""
    print(f"\n📄 Input ({len(raw_text)} chars): {raw_text[:80]}...")

    step1 = cleaner.run(raw_text)
    print(f"  🧹 After Cleaner: {step1[:60]}...")

    step2 = summariser.run(step1)
    print(f"  📝 After Summariser: {step2[:60]}...")

    step3 = categoriser.run(step2)
    print(f"  🏷️  After Categoriser: {step3}")

    step4 = formatter.run(f"Summary:\n{step2}\n\nCategory info: {step3}")
    return step4


RAW_TEXT = """
so basically like llms r really cool n all but they cant remember stuff
between conversations lol. which is kinda a big problem rite? like u tell it
ur name n then in the next chat its like hi how can i help u today smh.
mem0 and other tools r trying 2 fix this by giving LLMs persistent memory
so they can b more personal n stuff. pretty interesting tbh
"""

formatted = run_pipeline(RAW_TEXT)
print(f"\n✨ FINAL OUTPUT:\n{formatted}")


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN 3: DEBATE / CRITIC LOOP
# A writer produces content, a critic reviews it, the writer improves.
# Repeats until quality threshold is reached.
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*65)
print("PATTERN 3: DEBATE / CRITIC LOOP")
print("  Writer → Critic → Writer → Critic → ... → Final")
print("="*65)


class CritiqueResult(BaseModel):
    score: int = Field(ge=1, le=10)
    passes: bool = Field(description="True if quality is acceptable (score >= 7)")
    specific_issues: list[str]
    suggestions: list[str]
    overall_feedback: str


writer2 = Agent(
    name="ContentWriter",
    system_prompt=(
        "You write clear, accurate, engaging technical content. "
        "When given critique, revise your work addressing ALL specific issues. "
        "Improve based on suggestions. Keep content concise (under 150 words)."
    ),
)

critic = Agent(
    name="QualityReviewer",
    system_prompt=(
        "You are a tough but fair quality reviewer. Score content 1-10 on: "
        "accuracy, clarity, completeness, and engagement. "
        "Score 7+ means it passes. Be specific about what needs improvement."
    ),
)


def run_debate(task: str, quality_threshold: int = 7, max_rounds: int = 3) -> dict:
    """Run writer-critic debate until quality passes or max rounds reached."""
    print(f"\n📝 Task: {task}\n")

    # Initial draft
    draft = writer2.run(task)
    print(f"  ✍️  Round 1 draft: {draft[:100]}...")

    history = []
    for round_num in range(1, max_rounds + 1):
        # Critic reviews
        critique_raw = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a tough but fair quality reviewer. "
                        "Score content 1-10 on accuracy, clarity, completeness, engagement. "
                        "Score 7+ means it passes."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Review this content:\n\n{draft}",
                },
            ],
            response_format=CritiqueResult,
        )
        critique: CritiqueResult = critique_raw.choices[0].message.parsed

        history.append({"round": round_num, "draft": draft, "critique": critique})
        print(f"\n  🔍 Round {round_num} critique: score={critique.score}/10 pass={critique.passes}")
        print(f"     Issues: {critique.specific_issues[:2]}")

        if critique.passes:
            print(f"  ✅ Quality threshold reached at round {round_num}!")
            break

        if round_num < max_rounds:
            # Writer revises
            revision_prompt = (
                f"Original task: {task}\n\n"
                f"Your previous draft:\n{draft}\n\n"
                f"Critic's score: {critique.score}/10\n"
                f"Issues to fix: {critique.specific_issues}\n"
                f"Suggestions: {critique.suggestions}\n\n"
                f"Please revise addressing ALL issues."
            )
            draft = writer2.run(revision_prompt)
            print(f"  ✍️  Round {round_num + 1} revision: {draft[:100]}...")

    return {
        "final_draft": draft,
        "rounds": len(history),
        "final_score": history[-1]["critique"].score if history else 0,
        "history": history,
    }


result = run_debate(
    task="Explain the concept of RAG (Retrieval-Augmented Generation) for a non-technical business audience.",
    quality_threshold=7,
    max_rounds=3,
)

print(f"\n📊 Debate complete: {result['rounds']} rounds, final score: {result['final_score']}/10")
print(f"\n🏆 FINAL DRAFT:\n{result['final_draft']}")


print("\n\n💡 Summary of Multi-Agent Patterns:")
print("  Orchestrator + Workers → delegate, parallelise, synthesise")
print("  Sequential Pipeline    → transform data through specialised stages")
print("  Debate / Critic Loop   → iterate until quality threshold is met")
print("\n  Real frameworks (LangGraph, AutoGen, CrewAI) add:")
print("  - State management between agents")
print("  - Automatic tool use")
print("  - Human-in-the-loop checkpoints")
print("  - Persistent memory across sessions")
