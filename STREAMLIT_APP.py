import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
import io

st.set_page_config(page_title="Audio Transcription App", layout="centered")
st.title("🎤 Audio Transcription App")

st.write("Upload a WAV or MP3 file and get the speech converted to text.")


def convert_to_wav(uploaded_file):
    """Converts uploaded audio (MP3/WAV) to WAV format and returns a temp file path"""
    try:
        audio_bytes = uploaded_file.read()
        audio_buffer = io.BytesIO(audio_bytes)

        if uploaded_file.type == "audio/mpeg":
            audio = AudioSegment.from_mp3(audio_buffer)
        else:
            audio = AudioSegment.from_wav(audio_buffer)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio.export(temp_file.name, format="wav")
        return temp_file.name

    except Exception as e:
        st.error(f"Error converting file: {e}")
        return None


def transcribe_audio(filename):
    """Transcribes WAV file using Google Speech Recognition"""
    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(filename) as source:
            st.info("Processing audio...")
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio)
        return text

    except sr.UnknownValueError:
        st.error("Could not understand the audio 😔")
    except sr.RequestError as e:
        st.error(f"Speech Recognition service error: {e}")
    except Exception as e:
        st.error(f"Error: {e}")

    return None


uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3"])

if uploaded_file:
    st.success("File uploaded successfully!")

    wav_path = convert_to_wav(uploaded_file)

    if wav_path:
        st.audio(wav_path, format="audio/wav")

        if st.button("Transcribe"):
            transcription = transcribe_audio(wav_path)

            if transcription:
                st.subheader("📝 Transcription Result")
                st.write(transcription)
