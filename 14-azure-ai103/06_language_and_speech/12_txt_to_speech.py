import azure.cognitiveservices.speech as speechsdk
import sys
from pathlib import Path

_start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _parent in [_start, *_start.parents]:
    if (_parent / "azure_config.py").exists():
        sys.path.insert(0, str(_parent))
        break

from azure_config import config

speech_config=speechsdk.SpeechConfig(
    subscription=config.speech_key,
    endpoint=config.speech_endpoint
)

speech_config.speech_synthesis_voice_name="en-US-JennyNeural"

audio_output=speechsdk.audio.AudioOutputConfig(
    filename="cloudxeus_support_message.wav"
)

text = """
Hello, and thank you for contacting CloudXeus Technology Services.
Your support request has been received.
One of our cloud support specialists will review the issue and contact you shortly.
"""

speech_synthesizer = speechsdk.SpeechSynthesizer(
    speech_config=speech_config,
    audio_config=audio_output
)

result=speech_synthesizer.speak_text_async(text).get()

# Check the result
if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
    print("Speech synthesis completed successfully.")
    print("Audio file saved as cloudxeus_support_message.wav")
