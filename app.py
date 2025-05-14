import streamlit as st
import soundfile as sf
import numpy as np
import tensorflow as tf
import pickle
import speech_recognition as sr
import os
import tempfile
import time  # Import the time module

# --- Helper Functions (Modified for Streamlit) ---
def save_audio(audio_data, filename, samplerate=16000):
    """Saves audio data to a .wav file."""
    if audio_data is not None and audio_data.size > 0:
        sf.write(filename, audio_data, samplerate, subtype='PCM_16')
        st.success(f"Audio saved to {filename}")
        return filename  # Return the filename for further processing
    else:
        st.error("No audio data to save.")
        return None

def transcribe_audio(filename):
    """Transcribes audio from a .wav file using SpeechRecognition."""
    if not os.path.exists(filename):
        st.error(f"Error: File not found at {filename}")
        return None

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(filename) as source:
            audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            st.subheader("Transcription:")
            st.write(text)
            return text
        except sr.UnknownValueError:
            st.error("Speech Recognition could not understand the audio.")
            return None
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")
            return None
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None

def save_model(model, filename):
    """Saves a TensorFlow or pickle model."""
    if filename.endswith('.h5'):
        model.save(filename)
        st.success(f"Model saved to {filename}")
    elif filename.endswith('.pkl'):
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        st.success(f"Model saved to {filename}")
    else:
        st.error("Unsupported file format. Use .h5 or .pkl.")



# --- Streamlit App ---
def main():
    st.title("Audio Transcriber")

    # --- Upload Only ---
    st.write("Upload an audio file (WAV) for transcription.")
    uploaded_file = st.file_uploader("Upload Audio", type=["wav"])
    audio_filename = None

    if uploaded_file is not None:
        try:
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
                temp_audio_file.write(uploaded_file.read())
                audio_filename = temp_audio_file.name
                st.success("File uploaded successfully!")

                # --- Model and Transcription (Conditional) ---
                if audio_filename:
                    # Example: Using a simple pre-trained model (for demonstration purposes)
                    model = tf.keras.Sequential([
                        tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)),
                        tf.keras.layers.Dense(1, activation='sigmoid')
                    ])
                    #transcribe
                    transcription = transcribe_audio(audio_filename)

                    # Save model
                    model_filename = "speech_model.h5"
                    save_model(model, model_filename)
                    st.write(f"Model saved to {model_filename}")

                    # Optionally, display the audio
                    st.audio(audio_filename, format="audio/wav")

        except Exception as e:
            st.error(f"Error processing file: {e}")
            audio_filename = None
