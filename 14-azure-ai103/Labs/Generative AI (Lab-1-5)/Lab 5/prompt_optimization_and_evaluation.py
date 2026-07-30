# =============================================================================
# LAB 5 (NEW — added to fill an AI-103 syllabus gap, not an original course
# file): "Optimize" — the AI-103 course map's generative-AI learning path ends
# with "optimize (prompt engineering / RAG / fine-tune)". Lab 4 in this folder
# already covers the RAG half (file-search tool). This lab covers PROMPT
# ENGINEERING (few-shot examples, explicit output format) plus the pattern
# this repo actually uses for "evaluation": an **LLM-as-judge** critique loop,
# not the `azure-ai-evaluation` SDK (that package isn't used anywhere in this
# repo - the real, working pattern, found in
# `03_conversations_and_evaluation/03_response_completeness.py`, is a second plain model
# call that grades the first one's answer).
#
# Fine-tuning is intentionally NOT included here: it's a long-running,
# asynchronous training job (upload data -> submit job -> wait, often
# minutes to hours) that doesn't fit a short interactive lab - see
# `06_large_language_models/03_llm_fine_tuning/` elsewhere in this repo for
# a hands-on fine-tuning walkthrough instead.
#
# Prerequisites: Azure AI Foundry project + deployed chat model + `az login`
# + .env with PROJECT_ENDPOINT and MODEL_DEPLOYMENT_NAME.
# =============================================================================

import os                      # Env vars, clear-screen command
import sys
from pathlib import Path

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _parent in [_start, *_start.parents]:
    if (_parent / "azure_config.py").exists():
        sys.path.insert(0, str(_parent))
        break

from azure_config import config                    # Centralized Azure credential/config module

from azure.ai.projects import AIProjectClient       # Talks to your Foundry project
from azure.identity import DefaultAzureCredential   # Authenticates using your `az login` session


# ----------------------------------------------------------------------------
# Part 1: prompt engineering — few-shot examples + an explicit output format
# ----------------------------------------------------------------------------
def few_shot_classify(openai_client, model_deployment, review_text):
    """Classify a product review's sentiment using a FEW-SHOT prompt.

    "Few-shot" means the prompt includes a couple of worked EXAMPLES of the
    input/output pattern before the real question - this steers the model
    toward a consistent answer format far more reliably than just asking
    "what's the sentiment?" with no examples (that's "zero-shot").
    """
    few_shot_prompt = f"""Classify the sentiment of a product review as exactly one word: Positive, Negative, or Mixed.

Example 1:
Review: "This blender is amazing, it crushes ice in seconds!"
Sentiment: Positive

Example 2:
Review: "The battery died after two days and support never replied."
Sentiment: Negative

Example 3:
Review: "Great sound quality, but the app keeps crashing on setup."
Sentiment: Mixed

Now classify this review:
Review: "{review_text}"
Sentiment:"""

    response = openai_client.responses.create(
        model=model_deployment,
        input=few_shot_prompt,
    )
    # .strip() removes any stray whitespace/newlines around the one-word answer
    return response.output_text.strip()


# ----------------------------------------------------------------------------
# Part 2: LLM-as-judge evaluation loop (draft -> critique -> regenerate)
# ----------------------------------------------------------------------------
def draft_then_judge(openai_client, model_deployment, question):
    """Generate an answer, then have a SECOND model call grade it for
    completeness, and regenerate once if the judge says it's incomplete.

    This is the exact pattern used in this repo's real, working
    `03_conversations_and_evaluation/03_response_completeness.py` - adapted here to work
    against a plain model deployment (no pre-existing agent required),
    so this lab has no extra portal setup prerequisites.
    """
    # Pass 1: draft answer
    # Pass 1: draft answer
    draft_response = openai_client.responses.create(
        model=model_deployment,
        input=question,
    )
    answer = draft_response.output_text
    print(f"\n[Draft answer]\n{answer}\n")

    # Pass 2: judge call - a plain prompt, the SAME model grading its own
    # sibling's answer. The judge is given a strict, easy-to-parse output
    # convention ("reply with only COMPLETE or MISSING") so the code below
    # can check the verdict with a simple string search instead of parsing
    # free-form text.
    # Pass 2: judge call
    critique_response = openai_client.responses.create(
        model=model_deployment,
        input=(
            "You are reviewing an AI assistant's answer for completeness.\n\n"
            f"Question: {question}\n\n"
            f"Answer: {answer}\n\n"
            "Does the answer fully address every part of the question, with no "
            "obvious gaps or missing details? Reply with only COMPLETE or MISSING."
        ),
    )
    critique = critique_response.output_text
    print(f"[Judge verdict] {critique.strip()}")

    # Pass 3: conditional regeneration - only spend a THIRD model call if the
    # judge actually flagged a problem, rather than always regenerating.
    # Pass 3: conditional regeneration
    if "MISSING" in critique.upper():
        print("\n[Judge flagged gaps - regenerating a more complete answer...]")
        improved_response = openai_client.responses.create(
            model=model_deployment,
            input=(
                f"Answer this question completely, making sure to address every part of it: {question}"
            ),
        )
        return improved_response.output_text
    else:
        return answer


def main():
    # Clear the console
    os.system('cls' if os.name == 'nt' else 'clear')

    try:
        project_endpoint = config.project_endpoint
        model_deployment = config.model_deployment

        project_client = AIProjectClient(
            endpoint=project_endpoint,
            credential=DefaultAzureCredential(),
        )
        openai_client = project_client.get_openai_client()

        print("=" * 60)
        print("PART 1: Few-shot prompt engineering")
        print("=" * 60)
        sample_review = "The screen is gorgeous but it overheats during video calls."
        sentiment = few_shot_classify(openai_client, model_deployment, sample_review)
        print(f'Review: "{sample_review}"')
        print(f"Classified sentiment: {sentiment}")

        print("\n" + "=" * 60)
        print("PART 2: LLM-as-judge evaluation loop")
        print("=" * 60)
        question = input("\nAsk a question to test the draft -> judge -> regenerate loop:\n> ")
        if question:
            final_answer = draft_then_judge(openai_client, model_deployment, question)
            print(f"\n[Final answer]\n{final_answer}")

    except Exception as ex:
        # Catch-all: print any error instead of crashing
        print(ex)


if __name__ == "__main__":
    main()
