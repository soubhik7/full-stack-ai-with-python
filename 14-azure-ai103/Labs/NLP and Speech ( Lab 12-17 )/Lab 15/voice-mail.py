# =============================================================================
# LAB 15: A menu-driven console app using the DEDICATED Azure AI Speech SDK
# (azure.cognitiveservices.speech) directly - not an LLM audio model like
# Lab 14. Simulates a simple voicemail system: record a greeting (text ->
# speech) and transcribe incoming messages (speech -> text).
#
# Needs a `messages/` folder next to this file (not included) containing
# .wav files to transcribe via option 2 - add a few short recordings
# yourself.
#
# Prerequisites: an Azure AI Speech resource inside your Foundry project +
# `az login` + .env with FOUNDRY_ENDPOINT (FOUNDRY_KEY is also read but not
# actually used for authentication in this version - see the note below).
# Requires: pip install azure-cognitiveservices-speech playsound3 (neither
# is in the root requirements.txt).
# =============================================================================

from dotenv import load_dotenv          # Loads .env file variables into the environment
import os                               # Env vars, clear-screen command, folder listing
from playsound3 import playsound        # Plays a .wav file through your speakers

# Import namespaces
# import namespaces
from azure.identity import DefaultAzureCredential          # Authenticates using your `az login` session
import azure.cognitiveservices.speech as speech_sdk          # The dedicated Azure AI Speech SDK


def main():
    try:
        # Clear the console
        os.system('cls' if os.name == 'nt' else 'clear')

        # Get Configuration Settings
        load_dotenv()
        foundry_endpoint = os.getenv('FOUNDRY_ENDPOINT')
        foundry_key = os.getenv('FOUNDRY_KEY')  # loaded but not passed to SpeechConfig below - this build uses Entra ID auth instead of a key (a leftover variable from an older, key-based version of this lab)

        # Create speech_config using Entra ID authentication
        # Create speech_config using Entra ID authentication
        #
        # `SpeechConfig` is the central object every Speech SDK operation
        # (recognizer, synthesizer) is built from - it holds the endpoint
        # and how to authenticate. Passing `token_credential=` here means
        # NO subscription key is used; your `az login` identity handles
        # auth instead.
        credential = DefaultAzureCredential()
        speech_config = speech_sdk.SpeechConfig(
            token_credential=credential,
            endpoint=foundry_endpoint)


        # Loop until user quits
        inputText = ""
        while inputText.lower() != "3":
            inputText = input("Choose an option:\n1: Record a greeting\n2: Transcribe messages\n3: Exit\n")
            if inputText != "3":
                if inputText == "1":
                    record_greeting(speech_config)
                elif inputText == "2":
                    transcribe_messages(speech_config)
                elif inputText == "3":
                    # Unreachable in practice: the outer `while` condition
                    # already exits the loop the moment inputText == "3",
                    # so this branch never actually runs - a harmless
                    # leftover from how the menu was originally written.
                    print("Exiting...")
                    return
                else:
                    print("Invalid option, please try again.")



    except Exception as ex:
        # Catch-all: print any error instead of crashing
        print(ex)

# record_greeting function
def record_greeting(speech_config):
    print("Recording greeting...")

    # Get greeting message from user
    # Despite the function's name, this doesn't record your microphone -
    # you TYPE the greeting text, and the Speech SDK synthesizes it into
    # spoken audio (text-to-speech), which is the "recording" that gets saved.
    greeting_message = input("Enter your greeting message: ")


    # Synthesize the greeting message to an audio file
    output_file = "greeting.wav"
    # AudioOutputConfig tells the synthesizer to write to a FILE instead of
    # playing through speakers directly.
    audio_config = speech_sdk.audio.AudioOutputConfig(filename=output_file)

    # Picks a specific neural voice for the synthesized speech - Azure AI
    # Speech offers many named voices; this one is a high-definition
    # English voice.
    speech_config.speech_synthesis_voice_name = "en-US-Serena:DragonHDLatestNeural"

    speech_synthesizer = speech_sdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    # `speak_text_async(...).get()` runs the synthesis and blocks until it's
    # done (the `.get()` waits for the async operation to complete).
    result = speech_synthesizer.speak_text_async(greeting_message).get()

    if result.reason == speech_sdk.ResultReason.SynthesizingAudioCompleted:
        print(f"Greeting recorded and saved to {output_file}")
        speech_synthesizer = None  # Release the synthesizer resources
    else:
        # cancellation = result.cancellation_details
        # print(f"Cancellation Reason: {cancellation.reason}")
        # print(f"Error Code: {cancellation.error_code}")
        # print(f"Error Details: {cancellation.error_details}")
        # (the three lines above are commented out in the original course
        # file - uncomment them if you need to debug WHY synthesis failed,
        # since `result.reason` alone doesn't say much)
        print("Error recording greeting: {}".format(result.reason))



# transcribe_messages function
def transcribe_messages(speech_config):
    print("Transcribing messages...")

    messages_folder = 'messages'
    for file_name in os.listdir(messages_folder):
        if file_name.endswith('.wav'):
            print(f"\nTranscribing {file_name}...")
            file_path = os.path.join(messages_folder, file_name)
            # Play the file first so you can hear it alongside the printed
            # transcription and judge accuracy.
            playsound(file_path)

            # Transcribe the audio file
            # Transcribe the audio file
            # AudioConfig(filename=...) tells the recognizer to read from a
            # FILE instead of a live microphone.
            audio_config = speech_sdk.audio.AudioConfig(filename=file_path)
            speech_recognizer = speech_sdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config
            )
            # `recognize_once_async()` listens for a single utterance and
            # stops - fine for short voicemail-style clips, but NOT suited
            # to long recordings (those need continuous recognition instead).
            result = speech_recognizer.recognize_once_async().get()
            if result.reason == speech_sdk.ResultReason.RecognizedSpeech:
                print(f"Transcription: {result.text}")
            else:
                print("Error transcribing message: {}".format(result.reason))




if __name__ == "__main__":
    main()
