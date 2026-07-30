# pip3 install azure.ai.transcription
from azure.ai.transcription import TranscriptionClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.transcription.models import TranscriptionContent, TranscriptionOptions
import sys
from pathlib import Path

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _parent in [_start, *_start.parents]:
    if (_parent / "azure_config.py").exists():
        sys.path.insert(0, str(_parent))
        break

from azure_config import config

client=TranscriptionClient(
    endpoint=config.speech_endpoint,
    credential=AzureKeyCredential(config.speech_key)
)

audio_path="conversation.wav"
with open(audio_path, "rb") as audio_file:
    options = TranscriptionOptions(locales=["en-US"])
    result=client.transcribe(TranscriptionContent(definition=options, audio=audio_file))

print(result.combined_phrases[0].text)