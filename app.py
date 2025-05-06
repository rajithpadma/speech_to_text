import streamlit as st
import whisper
import tempfile
import os
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings

# Whisper model loader
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# Save audio as WAV
def save_audio(frames, sample_rate):
    import wave
    audio_data = np.concatenate(frames, axis=1).flatten().astype(np.int16).tobytes()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        with wave.open(f, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)
        return f.name

# Audio Processor
class AudioProcessor:
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.frames.append(frame.to_ndarray())
        return frame

# UI Setup
st.set_page_config(page_title="🎙️ Whisper STT", page_icon="🎙️")
st.title("🎙️ Whisper Speech-to-Text App")
mode = st.radio("Choose Mode:", ["🎤 Live Recording", "📂 Upload Audio File"])

# Mode 1: Live Recording
if mode == "🎤 Live Recording":
    st.markdown("Use your mic to record audio.")
    ctx = webrtc_streamer(
        key="mic",
        mode=WebRtcMode.SENDONLY,
        client_settings=ClientSettings(
            media_stream_constraints={"audio": True, "video": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        ),
        audio_receiver_size=1024,
        audio_processor_factory=AudioProcessor,
    )

    if ctx.audio_processor and ctx.audio_processor.frames:
        st.success("✅ Audio recorded. Click to transcribe.")
        if st.button("🔍 Transcribe"):
            sample_rate = ctx.audio_receiver.get_audio_frame_rate()
            wav_path = save_audio(ctx.audio_processor.frames, sample_rate)
            st.audio(wav_path, format="audio/wav")
            with st.spinner("Transcribing..."):
                result = model.transcribe(wav_path)
            st.subheader("📝 Transcription:")
            st.write(result["text"])
            st.download_button("💾 Download Transcription", result["text"], "transcription.txt")
            os.remove(wav_path)

# Mode 2: Upload Audio File
elif mode == "📂 Upload Audio File":
    st.markdown("Upload a .wav, .mp3, or .m4a file.")
    uploaded_file = st.file_uploader("Choose file", type=["wav", "mp3", "m4a"])

    if uploaded_file:
        file_ext = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        st.audio(tmp_path, format="audio/wav")
        with st.spinner("Transcribing..."):
            result = model.transcribe(tmp_path)
        st.subheader("📝 Transcription:")
        st.write(result["text"])
        st.download_button("💾 Download Transcription", result["text"], "transcription.txt")
        os.remove(tmp_path)
