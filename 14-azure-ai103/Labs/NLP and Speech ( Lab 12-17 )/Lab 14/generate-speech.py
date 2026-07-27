# =============================================================================
# LAB 14 (part 1 of 2): Text-to-speech using an AZURE OPENAI AUDIO MODEL
# (an LLM-hosted TTS model, called through the `AzureOpenAI` client) -
# NOT the dedicated Azure AI Speech SDK (compare with Lab 15/16, which use
# azure.cognitiveservices.speech for the same kind of task).
#
# This script converts one fixed sentence to speech, saves it as an MP3
# file, and plays it back through your speakers.
#
# Prerequisites: an Azure OpenAI audio-capable deployment (e.g. a
# "gpt-4o-mini-tts"-style model) + `az login` + .env with MODEL_ENDPOINT and
# MODEL_NAME.
# Requires: pip install playsound3 (not in the root requirements.txt).
# =============================================================================

import os                              # Env vars, clear-screen command
from pathlib import Path               # Cross-platform file path handling
from playsound3 import playsound       # Plays an audio file through your system's default speakers
from dotenv import load_dotenv          # Loads .env file variables into the environment

# Import namespaces
# import namespaces
from openai import AzureOpenAI                                              # The Azure-specific OpenAI SDK client (handles Azure's URL/versioning conventions)
from azure.identity import DefaultAzureCredential, get_bearer_token_provider  # Entra ID auth helpers (no API key needed)


def main():
    try:
        # Clear the console
        os.system('cls' if os.name == 'nt' else 'clear')

        # Get Configuration Settings
        load_dotenv()
        endpoint=os.getenv("MODEL_ENDPOINT")
        model_deployment=os.getenv("MODEL_NAME")
        # Path(__file__).parent = this script's own folder, so the output
        # file always lands next to the script regardless of your current
        # working directory.
        speech_file_path = Path(__file__).parent / "speech.mp3"


        # Create the Azure OpenAI client
        # Create the Azure OpenAI client
        # Same "get a bearer token from your az login session" pattern used
        # throughout this repo - no API key stored anywhere.
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://ai.azure.com/.default"
        )

        client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider = token_provider,
            api_version="2025-03-01-preview"   # Azure OpenAI requires an explicit API version, unlike the plain OpenAI client used in Lab 3
        )


        # Generate speech and save to file
        # Generate speech and save to file
        #
        # `with_streaming_response.create(...)` streams the generated audio
        # bytes directly to a file as they're produced, rather than loading
        # the whole clip into memory first. `voice="alloy"` picks one of the
        # model's built-in voice presets; `instructions` lets you steer HOW
        # it's spoken (tone, pacing, emotion) in plain English.
        with client.audio.speech.with_streaming_response.create(
                    model=model_deployment,
                    voice="alloy",
                    input="My voice is my passport!",
                    instructions="Speak in a serious tone.",
                ) as response:
            response.stream_to_file(speech_file_path)


        # Play the generated speech file
        playsound(speech_file_path)

    except Exception as ex:
        # Catch-all: print any error instead of crashing
        print(ex)

if __name__ == "__main__":
    main()
