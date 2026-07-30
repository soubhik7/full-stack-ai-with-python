# pip3 install azure.cognitiveservices.speech
import azure.cognitiveservices.speech as speechsdk
import time
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

audio_config=speechsdk.audio.AudioConfig(use_default_microphone=True)
speech_recognizer=speechsdk.SpeechRecognizer(speech_config=speech_config,audio_config=audio_config)

done = False

def stop_cb(evt):
    speech_recognizer.stop_continuous_recognition()
    global done
    done = True

speech_recognizer.recognized.connect(lambda evt: print(evt.result.text))
speech_recognizer.session_stopped.connect(stop_cb)
speech_recognizer.canceled.connect(stop_cb)

speech_recognizer.start_continuous_recognition()
while not done:
    time.sleep(0.5)