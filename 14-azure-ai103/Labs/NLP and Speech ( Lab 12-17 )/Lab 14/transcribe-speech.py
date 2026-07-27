# =============================================================================
# LAB 14 (part 2 of 2): Speech-to-text using an AZURE OPENAI AUDIO MODEL
# (e.g. a Whisper/gpt-4o-transcribe-style deployment), the mirror-image of
# generate-speech.py in this same folder.
#
# ⚠️ This script expects a file named "speech.wav" next to it, which is NOT
# included in this repo. generate-speech.py (in this same folder) produces
# "speech.mp3" instead - so either supply your own "speech.wav" sample
# recording, or adjust the filename/extension below to point at the .mp3
# file that script creates (the transcription API accepts multiple audio
# formats, mp3 included).
#
# Prerequisites: an Azure OpenAI audio-capable transcription deployment +
# `az login` + .env with MODEL_ENDPOINT and MODEL_NAME.
# Requires: pip install playsound3 (not in the root requirements.txt).
# =============================================================================

import os                              # Env vars, clear-screen command
from pathlib import Path               # Cross-platform file path handling
from playsound3 import playsound       # Plays the audio file so you can hear what's being transcribed
from dotenv import load_dotenv          # Loads .env file variables into the environment

# Import namespaces
# import namespaces
from openai import AzureOpenAI                                              # The Azure-specific OpenAI SDK client
from azure.identity import DefaultAzureCredential, get_bearer_token_provider  # Entra ID auth helpers


def main():
    try:
        # Clear the console
        os.system('cls' if os.name == 'nt' else 'clear')

        # Get Configuration Settings
        load_dotenv()
        endpoint = os.getenv("MODEL_ENDPOINT")
        model_deployment = os.getenv("MODEL_NAME")
        # Looks for "speech.wav" in this script's own folder.
        file_path = Path(__file__).parent / "speech.wav"

        # Play the speech file
        # Lets you hear the audio you're about to send for transcription,
        # so you can compare it against the printed result afterward.
        playsound(file_path)

        # Create the Azure OpenAI client
        # Create the Azure OpenAI client
        # Same auth pattern as generate-speech.py: a bearer token derived
        # from your `az login` session instead of an API key.
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://ai.azure.com/.default"
        )

        client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider = token_provider,
            api_version="2025-03-01-preview"
        )


        # Call model to transcribe audio file
        # Call model to transcribe audio file
        #
        # `"rb"` = read binary mode, required because audio files are
        # binary data, not text. `response_format="text"` tells the API to
        # return a plain string instead of a structured JSON object with
        # timestamps/segments.
        audio_file = open(file_path, "rb")
        transcription = client.audio.transcriptions.create(
            model=model_deployment,
            file=audio_file,
            response_format="text"
        )

        print(transcription)




    except Exception as ex:
        # Catch-all: print any error instead of crashing (e.g. missing
        # speech.wav file)
        print(ex)


if __name__ == "__main__":
    main()
