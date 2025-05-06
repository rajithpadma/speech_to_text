import streamlit as st
import whisper
import tempfile
import os
import sounddevice as sd
import numpy as np
import scipy.io.wavfile

# Page config
st.set_page_config(page_title="Whisper Transcriber", page_icon="🎙️")
st.title("🎙️ Whisper Speech-to-Text Transcriber")

# Load Whisper model once
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# Mode selection
mode = st.radio("Choose Mode", ["🎧 Upload Audio File", "🎤 Record Live"])

def transcribe_audio(path):
    try:
        result = model.transcribe(path)
        st.success("✅ Transcription Complete!")
        st.subheader("📝 Transcription:")
        st.write(result["text"])

        st.download_button(
            label="💾 Download Transcription as .txt",
            data=result["text"],
            file_name="transcription.txt",
            mime="text/plain"
        )
    except Exception as e:
        st.error(f"⚠️ Error during transcription: {e}")

# Mode 1: Upload audio file
if mode == "🎧 Upload Audio File":
    uploaded_file = st.file_uploader("📂 Upload a .wav, .mp3, or .m4a file", type=["wav", "mp3", "m4a"])
    if uploaded_file is not None:
        file_ext = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        st.audio(tmp_path, format="audio/wav")
        st.info("⏳ Transcribing... please wait.")
        transcribe_audio(tmp_path)
        os.remove(tmp_path)

# Mode 2: Record live audio
elif mode == "🎤 Record Live":
    duration = st.slider("🎚️ Recording Duration (seconds)", min_value=1, max_value=30, value=5)
    if st.button("🔴 Start Recording"):
        st.info("Recording...")
        fs = 16000
        try:
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
            sd.wait()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                scipy.io.wavfile.write(tmp_file.name, fs, recording)
                tmp_path = tmp_file.name

            st.audio(tmp_path, format="audio/wav")
            st.info("⏳ Transcribing your recording...")
            transcribe_audio(tmp_path)
            os.remove(tmp_path)

        except Exception as e:
            st.error(f"🎙️ Recording failed: {e}")
