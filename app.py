import streamlit as st
import soundfile as sf
import numpy as np
import tensorflow as tf
import pickle
import speech_recognition as sr
import os
import tempfile
import time
import base64

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

    audio_filename = None

    if record_or_upload == "Record":
        st.write("Click the button below to start recording. Recording will stop after 5 seconds.")

        # HTML and JavaScript for recording
        audio_html = """
            <div id="audio-recording-container">
                <button id="start-recording" style="background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;">Start Recording</button>
                <button id="stop-recording" style="background-color: #f44336; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; display: none;">Stop Recording</button>
                <audio id="audio-preview" controls style="display: none;"></audio>
                <p id="recording-status" style="margin-top: 10px;"></p>
            </div>
            <script>
            const startRecordingButton = document.getElementById('start-recording');
            const stopRecordingButton = document.getElementById('stop-recording');
            const audioPreview = document.getElementById('audio-preview');
            const recordingStatus = document.getElementById('recording-status');
            let mediaRecorder;
            let audioChunks = [];

            startRecordingButton.addEventListener('click', () => {
                navigator.mediaDevices.getUserMedia({ audio: true })
                    .then(stream => {
                        mediaRecorder = new MediaRecorder(stream);
                        mediaRecorder.ondataavailable = event => {
                            audioChunks.push(event.data);
                        };
                        mediaRecorder.onstop = () => {
                            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                            const reader = new FileReader();
                            reader.onloadend = () => {
                                const base64data = reader.result.split(',')[1];
                                // Send base64 data to Streamlit
                                Streamlit.set({ audio_data: base64data },'audio_data');
                                audioChunks = [];
                                recordingStatus.textContent = 'Recording stopped. Data sent to server.';
                            };
                            reader.readAsDataURL(audioBlob);
                            audioPreview.src = URL.createObjectURL(audioBlob);
                            audioPreview.style.display = 'block';
                            stream.getTracks().forEach(track => track.stop()); // Stop the stream
                        };
                        audioChunks = [];
                        mediaRecorder.start();
                        startRecordingButton.style.display = 'none';
                        stopRecordingButton.style.display = 'inline-block';
                        recordingStatus.textContent = 'Recording...';
                    })
                    .catch(error => {
                        console.error('Error accessing microphone:', error);
                        recordingStatus.textContent = 'Error accessing microphone: ' + error.message;
                        Streamlit.set({ audio_error: error.message }, 'audio_error');
                    });
            });

            stopRecordingButton.addEventListener('click', () => {
                if (mediaRecorder) {
                    mediaRecorder.stop();
                    stopRecordingButton.style.display = 'none';
                    startRecordingButton.style.display = 'inline-block';

                }
            });
            </script>
            """
        # Display the recording buttons
        audio_component = st.components.v1.html(audio_html, height=200)

        # Get the audio data from the JavaScript
        audio_data = st.session_state.get('audio_data')
        audio_error = st.session_state.get('audio_error')

        if audio_error:
            st.error(f"Error during recording: {audio_error}")
            audio_filename = None

        if audio_data:
            try:
                # Decode the base64 data
                audio_bytes = base64.b64decode(audio_data)
                # Use a temp file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
                    temp_audio_file.write(audio_bytes)
                    temp_audio_file.flush()  # Make sure the data is written
                    audio_filename = temp_audio_file.name
                    st.success("Audio data received and saved!")
            except Exception as e:
                st.error(f"Error processing audio data: {e}")
                audio_filename = None
        else:
            audio_filename = None

    elif record_or_upload == "Upload":
        uploaded_file = st.file_uploader("Upload an audio file (WAV)", type=["wav"])
        if uploaded_file is not None:
            try:
                # Save the uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
                    temp_audio_file.write(uploaded_file.read())
                    audio_filename = temp_audio_file.name
                    st.success("File uploaded successfully!")
                #audio_data, samplerate = sf.read(audio_filename)
            except Exception as e:
                st.error(f"Error uploading/processing file: {e}")
                audio_filename = None
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
