import streamlit as st
import tempfile
import whisper
import streamlit_webrtc
from streamlit_webrtc import webrtc_streamer,WebRtcMode, ClientSettings
import av
import numpy as np
import wave
import os
import uuid

st.set_page_config(page_title="Whisper Transcriber", page_icon="🎙️")
st.title("🎙️ Real-Time Voice Recorder + Transcription (Whisper)")

# Load Whisper model once
@st.cache_resource
def load_model():
    return whisper.load_model("base")  # Use "small", "medium", "large" for higher accuracy

model = load_model()

# Save WAV from audio buffer
def save_wav(audio, filename, sample_rate):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(audio)

# Audio processor class to collect audio frames
class AudioProcessor:
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        self.frames.append(audio)
        return frame

st.info("Click 'Start' to begin recording your voice using your browser's microphone.")

ctx = webrtc_streamer(
    key="speech",
    mode=WebRtcMode.SENDONLY,
    in_audio=True,
    client_settings=ClientSettings(
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    ),
    audio_receiver_size=1024,
    audio_processor_factory=AudioProcessor,
)

# Process after recording stops
if ctx.state.playing:
    st.warning("🔴 Recording... Speak now.")
else:
    if ctx.audio_receiver:
        audio_processor = ctx.audio_processor
        if audio_processor and audio_processor.frames:
            try:
                # Combine frames into audio buffer
                audio_data = np.concatenate(audio_processor.frames, axis=1).flatten().astype(np.int16).tobytes()

                # Save to WAV
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    filename = f.name
                    sample_rate = ctx.audio_receiver.get_audio_frame_rate()
                    save_wav(audio_data, filename, sample_rate)

                st.success("✅ Audio recorded successfully.")
                st.audio(filename, format="audio/wav")

                # Transcribe
                st.info("🔍 Transcribing with Whisper...")
                result = model.transcribe(filename)
                st.subheader("📝 Transcription:")
                st.write(result["text"])

                os.remove(filename)

            except Exception as e:
                st.error(f"⚠️ Error processing audio: {e}")
        else:
            st.info("🟡 No audio recorded yet. Try recording again.")
