import streamlit as st
import sounddevice as sd
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
    st.title("Audio Recorder and Transcriber")

    # --- Record/Upload Selection ---
    record_or_upload = st.radio("Record or Upload Audio", ("Record", "Upload"))

    audio_data = None
    audio_filename = None  # Keep track of the audio file

    if record_or_upload == "Record":
        st.write("Click the button below to start recording. Recording will stop after 5 seconds.")
        if st.button("Start Recording"):
            # Use a temporary file to store the recording
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
                try:
                    # Recording logic
                    samplerate = 16000
                    duration = 5  # Record for 5 seconds
                    st.write(f"Recording for {duration} seconds...")
                    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
                    sd.wait()  # Wait until recording is finished
                    audio_data = recording
                    audio_filename = temp_audio_file.name #get the name of the temp file.
                    sf.write(audio_filename, audio_data, samplerate) #save the recorded data

                    st.success("Recording complete!")

                except Exception as e:
                    st.error(f"Error during recording: {e}")
                    audio_data = None
                    audio_filename = None

    elif record_or_upload == "Upload":
        uploaded_file = st.file_uploader("Upload an audio file (WAV)", type=["wav"])
        if uploaded_file is not None:
            try:
                # Save the uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
                    temp_audio_file.write(uploaded_file.read())
                    audio_filename = temp_audio_file.name  # Get the name of the temp file
                    audio_data, samplerate = sf.read(audio_filename) #read and process
            except Exception as e:
                st.error(f"Error uploading/processing file: {e}")
                audio_data = None
                audio_filename = None

    # --- Model and Transcription (Conditional) ---
    if audio_data is not None and audio_filename is not None: #check if we have audio data.
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

if __name__ == "__main__":
    main()
