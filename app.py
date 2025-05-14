import streamlit as st
import sounddevice as sd
import soundfile as sf
import numpy as np
import tensorflow as tf
import speech_recognition as sr
import os

# Load the pre-trained model
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# Function to record audio
def record_audio(samplerate=16000, duration=10):
    st.info("Recording... Speak now!")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    return audio_data

# Function to save audio to a .wav file
def save_audio(audio_data, filename, samplerate=16000):
    sf.write(filename, audio_data, samplerate, subtype='PCM_16')
    st.success(f"Audio saved to {filename}")

# Function to transcribe audio using Google Speech Recognition
def transcribe_audio(filename):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(filename) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.error("Speech Recognition could not understand the audio.")
        return None
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service: {e}")
        return None

# Streamlit app
st.title("Speech-to-Text and Classification App")
st.markdown("Upload an audio file or record your voice to transcribe text and process it using a pre-trained model.")

# Load the model
model_path = "speech_model.h5"
model = load_model(model_path)

# Tabs for audio input
tab1, tab2 = st.tabs(["Upload Audio", "Record Audio"])

with tab1:
    uploaded_file = st.file_uploader("Upload an audio file (wav, mp3)", type=["wav", "mp3"])
    if uploaded_file is not None:
        file_path = os.path.join("temp_audio.wav")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Audio file uploaded successfully.")

        if st.button("Transcribe Uploaded Audio"):
            transcription = transcribe_audio(file_path)
            if transcription:
                st.text_area("Transcription", transcription, height=100)

with tab2:
    if st.button("Start Recording"):
        duration = st.slider("Select recording duration (seconds)", 5, 60, 10)
        audio_data = record_audio(duration=duration)
        file_path = "recorded_audio.wav"
        save_audio(audio_data, file_path)

        if st.button("Transcribe Recorded Audio"):
            transcription = transcribe_audio(file_path)
            if transcription:
                st.text_area("Transcription", transcription, height=100)

# Placeholder for model processing
st.markdown("---")
st.header("Model Processing")
if st.button("Run Model on Transcription"):
    if transcription:
        # Example processing: Create dummy input for model
        dummy_input = np.random.rand(1, 10)
        prediction = model.predict(dummy_input)
        st.write(f"Model Output: {prediction}")
    else:
        st.error("No transcription available to process.")
