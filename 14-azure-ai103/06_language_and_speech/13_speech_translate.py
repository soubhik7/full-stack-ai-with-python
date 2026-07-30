import azure.cognitiveservices.speech as speechsdk
import sys
from pathlib import Path

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _parent in [_start, *_start.parents]:
    if (_parent / "azure_config.py").exists():
        sys.path.insert(0, str(_parent))
        break

from azure_config import config

translation_config = speechsdk.translation.SpeechTranslationConfig(
    subscription=config.speech_key,
    region=config.speech_region
)

translation_config.speech_recognition_language="en-US"
translation_config.add_target_language("fr")

audio_config=speechsdk.audio.AudioConfig(use_default_microphone=True)

translation_recognizer = speechsdk.translation.TranslationRecognizer(
    translation_config=translation_config,
    audio_config=audio_config
)

print("Listening...")

result=translation_recognizer.recognize_once_async().get()

if result.reason == speechsdk.ResultReason.TranslatedSpeech:
    print("\nOriginal text:")
    print(result.text)

    print("\nFrench translation:")
    print(result.translations["fr"])