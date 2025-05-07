import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import whisper
import numpy as np
import tempfile
import os
import wave

# Set page config
st.set_page_config(page_title="Whisper Speech-to-Text", page_icon="🎙️")
st.title("🎙️ Whisper Speech-to-Text App")
st.markdown("This app records or uploads audio and transcribes it using OpenAI's Whisper model.")

# Load Whisper model once
@st.cache_resource
def load_model():
    return whisper.load_model("base")  # Adjust size as needed ("small", "medium", "large")

model = load_model()

# Custom audio processor for recording audio
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame):
        audio_data = frame.to_ndarray(copy=True)
        self.frames.append(audio_data)
        return frame

# Sidebar for user options
st.sidebar.title("Options")
method = st.sidebar.radio("Select an option:", ["Record Audio", "Upload Audio File"])

if method == "Record Audio":
    st.subheader("🎤 Record Audio")
    ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
    )
    if ctx.audio_processor and st.button("🎙️ Save and Transcribe"):
        audio_frames = ctx.audio_processor.frames
        if len(audio_frames) == 0:
            st.warning("No audio recorded. Please record your voice and try again.")
        else:
            # Save audio frames to a temporary .wav file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name

            audio_data = np.concatenate(audio_frames, axis=0).astype(np.int16)
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_data.tobytes())

            st.audio(tmp_path, format="audio/wav")
            
            # Transcribe and display the result
            try:
                result = model.transcribe(tmp_path)
                st.subheader("📝 Transcription:")
                st.write(result["text"])
                st.download_button("💾 Download Transcript", result["text"], "transcription.txt")
            except Exception as e:
                st.error(f"❌ Transcription failed: {e}")
            finally:
                os.remove(tmp_path)  # Clean up temporary file

elif method == "Upload Audio File":
    st.subheader("📤 Upload Audio File")
    uploaded_file = st.file_uploader("Upload an audio file (wav/m4a/mp4):", type=["wav", "m4a", "mp4"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(uploaded_file.read())
        
        st.audio(tmp_path, format=f"audio/{uploaded_file.type.split('/')[-1]}")
        
        # Transcribe and display the result
        try:
            result = model.transcribe(tmp_path)
            st.subheader("📝 Transcription:")
            st.write(result["text"])
            st.download_button("💾 Download Transcript", result["text"], "transcription.txt")
        except Exception as e:
            st.error(f"❌ Transcription failed: {e}")
        finally:
            os.remove(tmp_path)  # Clean up temporary file
