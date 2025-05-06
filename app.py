import streamlit as st
import whisper
import tempfile
import os
import numpy as np
import wave
import sounddevice as sd

# Set page config
st.set_page_config(page_title="Whisper Speech-to-Text", page_icon="🎙️")
st.title("🎙️ Whisper Speech-to-Text App")

st.markdown("Choose a mode to transcribe speech using OpenAI's Whisper model.")

# Load Whisper model once
@st.cache_resource
def load_model():
    return whisper.load_model("base")  # Use "small", "medium", "large" for more accuracy

model = load_model()

# Function to record audio
def record_audio(duration, fs=16000):
    st.info(f"⏳ Recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    st.success("✅ Recording finished!")
    return audio_data.flatten()

# Function to save audio to wav
def save_audio_to_wav(audio_data, filename, fs=16000):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(2)  # 2 bytes per sample
        wf.setframerate(fs)  # Sample rate (16000 Hz)
        wf.writeframes(audio_data)
    st.audio(filename)  # Display audio player

# Select Mode
mode = st.radio("Select Mode:", ["📁 Upload Audio File", "🎤 Record Live"])

# Mode 1: Upload
if mode == "📁 Upload Audio File":
    uploaded_file = st.file_uploader("Upload a .wav, .mp3, or .m4a file", type=["wav", "mp3", "m4a"])

    if uploaded_file:
        file_ext = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.audio(tmp_path)
        st.info("⏳ Transcribing...")

        try:
            result = model.transcribe(tmp_path)
            st.subheader("📝 Transcription:")
            st.write(result["text"])
            st.download_button("💾 Download Transcript", result["text"], "transcription.txt")
        except Exception as e:
            st.error(f"❌ Transcription failed: {e}")
        finally:
            os.remove(tmp_path)

# Mode 2: Record with Mic
elif mode == "🎤 Record Live":
    st.info("🎙️ Press the button below to start recording.")

    duration = st.slider("Set the recording duration (seconds)", 1, 60, 5)

    if st.button("Start Recording"):
        audio_data = record_audio(duration)

        # Save the recorded audio as a .wav file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            audio_filename = f.name
        save_audio_to_wav(audio_data, audio_filename)

        st.info("⏳ Transcribing...")

        try:
            result = model.transcribe(audio_filename)
            st.subheader("📝 Transcription:")
            st.write(result["text"])

            # Provide download link for the transcription
            st.download_button("💾 Download Transcript", result["text"], "transcription.txt")
        except Exception as e:
            st.error(f"❌ Transcription failed: {e}")
        finally:
            os.remove(audio_filename)
