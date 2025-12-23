import streamlit as st
from pydub import AudioSegment
import io
import wave

# Function to record audio and save as .wav
def record_audio():
    audio_file = st.file_uploader("Upload your voice (in mp3 or wav format)", type=["mp3", "wav"])
    if audio_file is not None:
        # Read audio file
        audio_data = audio_file.read()

        # Convert to WAV format if it's not already
        if audio_file.type == "audio/mpeg":  # MP3 format
            audio = AudioSegment.from_mp3(io.BytesIO(audio_data))
        else:  # WAV format
            audio = AudioSegment.from_wav(io.BytesIO(audio_data))

        # Save as WAV file
        wav_file_name = "recorded_audio.wav"
        audio.export(wav_file_name, format="wav")
        st.success(f"Audio saved as {wav_file_name}")
        return wav_file_name

st.title("Voice Recorder App")

if st.button("Record"):
    wav_file = record_audio()
    if wav_file:
        with open(wav_file, "rb") as f:
            st.download_button("Download Audio", f, file_name=wav_file)
