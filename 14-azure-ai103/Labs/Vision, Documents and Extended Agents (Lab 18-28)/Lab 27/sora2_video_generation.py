# =============================================================================
# LAB 27 (NEW — added to fill an AI-103 syllabus gap, not an original course
# file).
#
# ⚠️⚠️⚠️ PREVIEW / BEST-EFFORT — UNVERIFIED CODE ⚠️⚠️⚠️
# EXAM_NOTES.md lists Sora 2 video generation in Foundry as an async,
# submit-then-poll job (see "12 · Image & video generation": *"text prompt
# (+ reference media) -> video; async job (submit-then-poll)"*). No video-
# generation code exists anywhere in this repo (only image generation, in
# `05. Section Code/` and this folder's Lab 22). The call shape below
# (`client.videos.create(...)` / `client.videos.retrieve(...)` /
# `client.videos.download_content(...)`) is this lab's best-effort guess,
# modeled on OpenAI's public video-generation API pattern extended to an
# Azure OpenAI endpoint the same way Lab 22's image generation is - it is
# NOT confirmed to be available or to match this exactly on your Azure
# OpenAI resource/API version. Availability and the exact method names may
# differ; check the current Azure OpenAI / Foundry video-generation docs
# before relying on this.
#
# Prerequisites (once verified): an Azure OpenAI resource with a Sora-
# family video model deployed + `az login` + .env with
# AZURE_OPENAI_ENDPOINT and VIDEO_DEPLOYMENT_NAME.
# =============================================================================

import os
import time
from pathlib import Path
from dotenv import load_dotenv

from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider


def main():
    os.system('cls' if os.name == 'nt' else 'clear')

    load_dotenv()
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    video_deployment = os.getenv("VIDEO_DEPLOYMENT_NAME", "sora-2")

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://ai.azure.com/.default"
    )
    client = OpenAI(base_url=azure_openai_endpoint, api_key=token_provider)

    print("Concept: video generation is an ASYNC job, unlike image generation")
    print("(Lab 22), which returns the finished image in one call. You submit a")
    print("job, then poll its status until it's done, then download the result.\n")

    try:
        # --- Step 1: submit the generation job ---
        # ⚠️ Best-effort guessed call shape.
        print("Step 1/3: Submitting the video generation job (best-effort call)...")
        video_job = client.videos.create(
            model=video_deployment,
            prompt="A time-lapse of clouds moving over a mountain range at sunrise.",
            size="1280x720",
            seconds="4",   # video generation charges/limits by duration, unlike a fixed image size
        )
        print(f"  Job submitted: id={video_job.id}, status={video_job.status}")

        # --- Step 2: poll until the job finishes ---
        print("\nStep 2/3: Polling until the job completes...")
        while video_job.status in ("queued", "processing", "in_progress"):
            time.sleep(5)
            video_job = client.videos.retrieve(video_job.id)
            print(f"  status={video_job.status}")

        if video_job.status != "completed":
            print(f"Job ended with status '{video_job.status}' - nothing to download.")
            return

        # --- Step 3: download the finished video ---
        print("\nStep 3/3: Downloading the finished video...")
        content = client.videos.download_content(video_job.id, variant="video")
        output_path = Path(__file__).parent / "generated_video.mp4"
        content.write_to_file(output_path)
        print(f"  Saved: {output_path}")

    except Exception as ex:
        print(f"\n(Expected in this preview lab if Sora isn't available on your resource, or")
        print(f"if the real API shape differs from this best-effort guess: {ex})")
        print("\nCheck the current Azure OpenAI / Foundry video-generation documentation")
        print("for the confirmed client methods, then adjust this script accordingly.")


if __name__ == "__main__":
    main()
