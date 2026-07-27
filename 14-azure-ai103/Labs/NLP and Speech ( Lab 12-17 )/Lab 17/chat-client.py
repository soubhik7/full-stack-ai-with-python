# =============================================================================
# LAB 17: The most advanced file in this Labs folder - a FULL-DUPLEX,
# REAL-TIME voice conversation with a Foundry agent, using Azure AI
# VoiceLive (a WebSocket-based streaming voice service) and your computer's
# microphone/speakers via PyAudio. "Full-duplex" means you can be listening
# and speaking (interrupting) at the same time, like a real phone call -
# unlike Lab 14/15/16 which do one-shot record-then-transcribe or
# type-then-synthesize.
#
# Read this file LAST, after the other agent/speech labs - it combines
# real-time audio I/O with the agent-conversation concepts from earlier labs.
#
# Needs a pre-built portal agent registered with VoiceLive (set
# AZURE_VOICELIVE_ENDPOINT, AZURE_VOICELIVE_AGENT_ID, and
# AZURE_VOICELIVE_PROJECT_NAME in your .env) + `az login`.
#
# Requires a working microphone and speakers, plus:
#   pip install azure-ai-voicelive pyaudio
# On macOS, PyAudio needs the PortAudio system library first:
#   brew install portaudio && pip install pyaudio
# =============================================================================

import os                          # Env vars, clear-screen command
import asyncio                     # VoiceLive streams events continuously, so this whole script is async
import base64                      # Encodes/decodes raw audio bytes as base64 text for the WebSocket protocol
import queue                       # A thread-safe queue.Queue buffers incoming audio between the network callback and the audio-playback callback
from dotenv import load_dotenv      # Loads .env file variables into the environment
import pyaudio                      # Cross-platform library for microphone input / speaker output

# import namespaces
# import namespaces
from azure.identity.aio import AzureCliCredential   # The ASYNC version of Azure CLI auth (".aio" module)
from azure.ai.voicelive.aio import connect           # Opens an async WebSocket connection to the VoiceLive service
from azure.ai.voicelive.models import (
    InputAudioFormat,               # Describes the audio encoding WE send (e.g. 16-bit PCM)
    Modality,                       # Which kinds of content the session should exchange - text, audio, or both
    OutputAudioFormat,              # Describes the audio encoding the SERVICE sends back
    RequestSession,                 # The full session configuration object sent to VoiceLive
    ServerEventType,                # An enum of every event type the server can send us
    AudioNoiseReduction,             # Configures background-noise suppression on the input audio
    AudioEchoCancellation,           # Prevents the mic from picking up the speaker's own output ("hearing itself")
    AzureSemanticVadMultilingual,    # Voice Activity Detection - automatically figures out when you've started/stopped talking
    AgentConfig                     # Identifies WHICH Foundry agent this VoiceLive session should talk to
)


def main():
    """Main entry point."""

    try:
        # Clear the console
        os.system('cls' if os.name == 'nt' else 'clear')

        # Get required configuration from environment variables
        load_dotenv()
        endpoint = os.environ.get("AZURE_VOICELIVE_ENDPOINT")
        agent_name = os.environ.get("AZURE_VOICELIVE_AGENT_ID")
        project_name = os.environ.get("AZURE_VOICELIVE_PROJECT_NAME")
        # Tells VoiceLive which portal-created agent (and which Foundry
        # project it lives in) should power this conversation.
        agent_config = AgentConfig({ "agent_name": agent_name, "project_name": project_name })


        # Create credential for authentication
        credential = AzureCliCredential()

        # Create and start the voice assistant
        assistant = VoiceAssistant(
            endpoint=endpoint,
            credential=credential,
            agent_config = agent_config
        )

        # Run the assistant
        try:
            # asyncio.run(...) is the standard way to kick off a top-level
            # async function from ordinary (synchronous) code.
            asyncio.run(assistant.start())
        except KeyboardInterrupt:
            # Exit if the user enters CTRL+C
            print("\n👋 Goodbye!")


    except Exception as e:
        print(f"❌ An error occurred: {e}")


# VoiceAssistant class - main coordinator for the voice agent
class VoiceAssistant:
    """
    Main voice assistant that coordinates the conversation flow.

    This class demonstrates the essential pattern for Azure VoiceLive:
    1. Connect to the service
    2. Configure the session
    3. Start audio capture/playback
    4. Process events from the service
    """

    def __init__(self, endpoint, credential, agent_config):
        # Just stores the settings passed in from main() - the real work
        # happens in start(), once this object is actually run.
        self.endpoint = endpoint
        self.credential = credential
        self.agent_config = agent_config
    async def start(self):
        """Start the voice assistant."""
        print("\n" + "=" * 60)
        print("🎙️  AZURE VOICELIVE VOICE AGENT")
        print("=" * 60)

        # Add your code in this try block!
        try:
            # STEP 1: Connect Azure VoiceLive to the agent
            # STEP 1: Connect Azure VoiceLive to the agent
            #
            # `connect(...)` opens an async WebSocket to the VoiceLive
            # service and, used as `async with`, automatically closes it
            # when this block exits (even on error).
            async with connect(
                endpoint=self.endpoint,
                credential=self.credential,
                api_version="2026-01-01-preview",
                agent_config=self.agent_config
            ) as connection:
                self.connection = connection

                # STEP 2: Initialize audio processor
                # STEP 2: Initialize audio processor
                # AudioProcessor (defined below) owns the actual microphone
                # and speaker streams - it's created here once the
                # connection exists, since it needs to send/receive audio
                # through it.
                self.audio_processor = AudioProcessor(connection)

                # STEP 3: Configure the session
                # STEP 3: Configure the session
                await self.setup_session()

                # STEP 4: Start audio systems
                # STEP 4: Start audio systems
                # Speaker output is started immediately; microphone capture
                # only starts once the server confirms the session is ready
                # (see handle_event's SESSION_UPDATED branch below) - this
                # avoids sending audio before VoiceLive is ready to receive it.
                self.audio_processor.start_playback()

                print("\n✅ Ready! Start speaking...")
                print("Press Ctrl+C to exit\n")

                # STEP 5: Process events
                # STEP 5: Process events
                # This call runs until the connection closes or the program
                # is interrupted - it's the main "listen for whatever
                # happens next" loop.
                await self.process_events()


        finally:
            # Always release microphone/speaker resources on the way out,
            # whether start() finished normally or an exception/KeyboardInterrupt
            # cut it short.
            if hasattr(self, 'audio_processor'):
                self.audio_processor.shutdown()

    async def setup_session(self):
        """Configure the session with audio settings."""

        # RequestSession bundles every setting that controls how this
        # real-time conversation behaves.
        session_config = RequestSession(
            # Enable both text and audio
            # The agent can produce both a text transcript AND spoken audio
            # for its replies - Modality.TEXT lets us show
            # transcripts on screen while also hearing the voice.
            modalities=[Modality.TEXT, Modality.AUDIO],

            # Audio format (16-bit PCM at 24kHz)
            # PCM16 is uncompressed, raw audio - simple to encode/decode in
            # real time, at the cost of using more bandwidth than a
            # compressed codec.
            input_audio_format=InputAudioFormat.PCM16,
            output_audio_format=OutputAudioFormat.PCM16,

            # Voice activity detection (when to detect speech)
            # Instead of you pressing a button to say "I'm done talking",
            # VoiceLive listens for natural pauses/silence and figures out
            # on its own when your turn has ended - "semantic" VAD is smart
            # enough to not cut you off mid-sentence just because you paused
            # briefly.
            turn_detection=AzureSemanticVadMultilingual(),

            # Prevent echo from speaker feedback
            # Without this, the microphone would pick up the agent's own
            # voice coming out of your speakers and think YOU were talking.
            input_audio_echo_cancellation=AudioEchoCancellation(),

            # Reduce background noise
            input_audio_noise_reduction=AudioNoiseReduction(type="azure_deep_noise_suppression")
        )

        # Sends this configuration to the server over the open connection -
        # the server replies with a SESSION_UPDATED event once it's applied
        # (handled in handle_event below).
        await self.connection.session.update(session=session_config)
        print("⚙️  Session configured")

    async def process_events(self):
        """Process events from the VoiceLive service."""

        # Listen for events from the service
        # `async for event in self.connection:` continuously awaits the
        # next message from the WebSocket - this loop runs for the entire
        # lifetime of the conversation, dispatching each event as it arrives.
        async for event in self.connection:
            await self.handle_event(event)

    async def handle_event(self, event):
        """Handle different event types from the service."""

        # Session is ready - start capturing audio
        # Only NOW (after the server confirms our session settings were
        # applied) do we open the microphone - starting earlier could send
        # audio the server isn't ready to process yet.
        if event.type == ServerEventType.SESSION_UPDATED:
            print(f"📡 Connected to agent: {event.session.agent.name}")
            self.audio_processor.start_capture()

        # User speech was transcribed
        # The server ran speech-to-text on what you said and is reporting
        # the resulting text - just for display, doesn't affect the flow.
        elif event.type == ServerEventType.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED:
            print(f'👤 You: {event.get("transcript", "")}')

        # Agent is responding with audio transcript
        # The text version of what the agent is (or will be) saying out
        # loud - printed alongside the actual audio for readability.
        elif event.type == ServerEventType.RESPONSE_AUDIO_TRANSCRIPT_DONE:
            print(f'🤖 Agent: {event.get("transcript", "")}')

        # User started speaking (interrupt any playing audio)
        # This is the "barge-in" / interruption feature: if the agent is
        # mid-sentence and you start talking, immediately stop whatever
        # audio is queued for playback so your voice isn't talked over.
        elif event.type == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED:
            self.audio_processor.clear_playback_queue()
            print("🎤 Listening...")

        # User stopped speaking
        # VAD detected you've finished your turn - the agent will now start
        # formulating a response.
        elif event.type == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STOPPED:
            print("🤔 Thinking...")

        # Receiving audio response chunks
        # The agent's spoken reply arrives in small pieces ("deltas") as
        # it's generated - each chunk gets handed to the playback queue
        # immediately so audio starts playing before the full reply is done
        # (this is what makes it feel real-time rather than laggy).
        elif event.type == ServerEventType.RESPONSE_AUDIO_DELTA:
            self.audio_processor.queue_audio(event.delta)

        # Audio response complete
        elif event.type == ServerEventType.RESPONSE_AUDIO_DONE:
            print("✓ Response complete\n")

        # Handle errors
        elif event.type == ServerEventType.ERROR:
            print(f"❌ Error: {event.error.message}")


# AudioProcessor class - handles microphone input and speaker output using PyAudio
class AudioProcessor:
    """
    Handles audio input (microphone) and output (speakers).

    Key responsibilities:
    - Capture audio from microphone and send to VoiceLive
    - Receive audio from VoiceLive and play through speakers
    """

    def __init__(self, connection):
        self.connection = connection
        # PyAudio() is the entry point to the system's audio hardware -
        # everything below (input/output streams) is opened through it.
        self.audio = pyaudio.PyAudio()

        # Audio settings: 24kHz, 16-bit PCM, mono
        # These MUST match the format configured in setup_session() above
        # (InputAudioFormat.PCM16 / OutputAudioFormat.PCM16) or the audio
        # will sound garbled/wrong-speed.
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 24000
        self.chunk_size = 1200  # 50ms chunks

        # Streams for input and output
        # Not created yet - populated later by start_capture()/start_playback().
        self.input_stream = None
        self.output_stream = None
        # A thread-safe queue for audio bytes waiting to be played - PyAudio
        # callbacks run on a SEPARATE thread from the main asyncio event
        # loop, so a plain Python list wouldn't be safe to share between them.
        self.playback_queue = queue.Queue()

    def start_capture(self):
        """Start capturing audio from the microphone."""

        def capture_callback(in_data, frame_count, time_info, status):
            # PyAudio calls this function automatically, on its own
            # background thread, every time a new chunk of microphone audio
            # is available (`in_data`).
            #
            # Convert audio to base64 and send to VoiceLive
            # Raw audio bytes can't go directly into a JSON/WebSocket
            # message, so we base64-encode them into plain text first.
            audio_base64 = base64.b64encode(in_data).decode("utf-8")
            # `asyncio.run_coroutine_threadsafe` is the bridge between
            # PyAudio's background thread and the main asyncio event loop:
            # it safely schedules the async `append(...)` call to run on
            # the event loop from this non-async callback thread.
            asyncio.run_coroutine_threadsafe(
                self.connection.input_audio_buffer.append(audio=audio_base64),
                self.loop
            )
            # Returning (None, paContinue) tells PyAudio "keep the stream
            # running, I have nothing to give back for output" (this is an
            # input-only stream).
            return (None, pyaudio.paContinue)

        # Store event loop for use in callback thread
        # Captured now (while we're still on the main async thread) so the
        # callback above - running on PyAudio's separate thread - has a
        # valid reference to schedule work back onto.
        self.loop = asyncio.get_event_loop()

        self.input_stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=capture_callback   # PyAudio runs this every ~50ms with new mic data
        )
        print("🎤 Microphone started")

    def start_playback(self):
        """Start audio playback system."""

        # `remaining` holds any leftover audio bytes from the previous
        # callback that didn't neatly divide into a full chunk - `nonlocal`
        # lets the inner function modify this outer variable across calls.
        remaining = bytes()

        def playback_callback(in_data, frame_count, time_info, status):
            # PyAudio calls this automatically whenever it needs MORE audio
            # to send to the speakers - `frame_count` tells us exactly how
            # many audio frames (samples) it needs right now.
            nonlocal remaining

            # Calculate bytes needed
            bytes_needed = frame_count * pyaudio.get_sample_size(pyaudio.paInt16)
            output = remaining[:bytes_needed]
            remaining = remaining[bytes_needed:]

            # Get more audio from queue if needed
            # Keep pulling chunks off the queue (filled by queue_audio(),
            # called from the VoiceLive event handler) until we have enough
            # bytes to satisfy this callback.
            while len(output) < bytes_needed:
                try:
                    audio_data = self.playback_queue.get_nowait()
                    if audio_data is None:  # End signal
                        break
                    output += audio_data
                except queue.Empty:
                    # No audio ready yet - pad with silence (zero bytes)
                    # rather than leaving a gap, so playback doesn't glitch
                    # or crash waiting for data.
                    output += bytes(bytes_needed - len(output))
                    break

            # Keep any extra for next callback
            # If we pulled a chunk from the queue that was bigger than what
            # this callback needed, save the leftover for the NEXT callback
            # instead of throwing it away.
            if len(output) > bytes_needed:
                remaining = output[bytes_needed:]
                output = output[:bytes_needed]

            # Returning the audio bytes plus paContinue tells PyAudio "play
            # this, then call me again for more."
            return (output, pyaudio.paContinue)

        self.output_stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            output=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=playback_callback
        )
        print("🔊 Speakers ready")

    def queue_audio(self, audio_data):
        """Add audio data to the playback queue."""
        # Called from handle_event() whenever a RESPONSE_AUDIO_DELTA chunk
        # arrives from the server - hands it off to the playback thread via
        # the thread-safe queue.
        self.playback_queue.put(audio_data)

    def clear_playback_queue(self):
        """Clear any pending audio (used when user interrupts)."""
        # Drains every pending chunk without blocking - used when the user
        # starts talking, so the agent's queued reply doesn't keep playing
        # over them (see INPUT_AUDIO_BUFFER_SPEECH_STARTED in handle_event).
        while not self.playback_queue.empty():
            try:
                self.playback_queue.get_nowait()
            except queue.Empty:
                break

    def shutdown(self):
        """Clean up audio resources."""
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()

        if self.output_stream:
            self.playback_queue.put(None)  # Signal end
            self.output_stream.stop_stream()
            self.output_stream.close()

        # Releases PyAudio's hold on the system's audio hardware entirely -
        # without this, subsequent runs of this script (or other audio
        # apps) could have trouble accessing the microphone/speakers.
        self.audio.terminate()
        print("🔇 Audio stopped")





if __name__ == "__main__":
    main()
