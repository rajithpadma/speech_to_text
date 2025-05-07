import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import whisper
import numpy as np
import tempfile
import os

# Set page config
st.set_page_config(page_title="Whisper Speech-to-Text", page_icon="🎙️")
st.title("🎙️ Whisper Speech-to-Text App")

st.markdown("This app records audio from your microphone and transcribes it using OpenAI's Whisper model.")

# Load Whisper model once
@st.cache_resource
def load_model():
    return whisper.load_model("base")  # Use "small", "medium", "large" for more accuracy

model = load_model()

# Custom audio processor for recording audio
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame):
        audio_data = frame.to_ndarray(copy=True)
        self.frames.append(audio_data)
        return frame

# WebRTC Streamer for recording audio
ctx = webrtc_streamer(
    key="speech-to-text",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

# Process the recorded audio
if ctx.audio_processor and st.button("🎙️ Transcribe Audio"):
    audio_frames = ctx.audio_processor.frames
    if len(audio_frames) == 0:
        st.warning("No audio recorded. Please record your voice and try again.")
    else:
        # Save recorded audio to a temporary .wav file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        # Combine frames and save as WAV
        audio_data = np.concatenate(audio_frames, axis=0).astype(np.int16)
        with open(tmp_path, "wb") as f:
            import wave
            with wave.open(f, "wb") as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(16000)  # Sample rate
                wf.writeframes(audio_data.tobytes())

        st.audio(tmp_path, format="audio/wav")

        # Transcribe the audio with Whisper
        try:
            result = model.transcribe(tmp_path)
            st.subheader("📝 Transcription:")
            st.write(result["text"])
            st.download_button("💾 Download Transcript", result["text"], "transcription.txt")
        except Exception as e:
            st.error(f"❌ Transcription failed: {e}")
        finally:
            os.remove(tmp_path)  # Clean up the temporary file
