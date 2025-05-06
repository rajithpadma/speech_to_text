import streamlit as st
import whisper
import tempfile
import os
import numpy as np
import base64
import io
from io import BytesIO
import wave
from streamlit.components.v1 import html

# Set page config
st.set_page_config(page_title="Whisper Speech-to-Text", page_icon="🎙️")
st.title("🎙️ Whisper Speech-to-Text App")

st.markdown("Choose a mode to transcribe speech using OpenAI's Whisper model.")

# Load Whisper model once
@st.cache_resource
def load_model():
    return whisper.load_model("base")  # Use "small", "medium", "large" for more accuracy

model = load_model()

# Function to convert audio to wav
def save_audio_to_wav(audio_data, filename):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(2)  # 2 bytes per sample
        wf.setframerate(16000)  # Sample rate (16000 Hz)
        wf.writeframes(audio_data)
    st.audio(filename)  # Display audio player

# Function to handle base64 audio input from the browser
def handle_audio_upload(audio_base64):
    audio_data = base64.b64decode(audio_base64.split(",")[1])  # Decode the base64 audio
    audio_filename = "temp_audio.wav"
    
    # Save the audio to a .wav file
    with open(audio_filename, "wb") as f:
        f.write(audio_data)

    st.audio(audio_filename)  # Display the audio player for the uploaded file
    
    # Transcribe the audio with Whisper
    try:
        result = model.transcribe(audio_filename)
        st.subheader("📝 Transcription:")
        st.write(result["text"])
        
        # Provide the download button for the transcription
        st.download_button("💾 Download Transcript", result["text"], "transcription.txt")
    except Exception as e:
        st.error(f"❌ Transcription failed: {e}")
    finally:
        os.remove(audio_filename)  # Clean up the temp file

# Custom HTML to record audio in the browser using JavaScript
html_code = """
    <script>
        var audioRecorder;
        var audioChunks = [];

        // Setup the audio recorder
        async function startRecording() {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const recorder = new MediaRecorder(stream);
            recorder.ondataavailable = event => audioChunks.push(event.data);
            recorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const audioUrl = URL.createObjectURL(audioBlob);
                const reader = new FileReader();
                reader.onloadend = function() {
                    var base64Audio = reader.result;
                    const downloadLink = document.createElement('a');
                    downloadLink.href = audioUrl;
                    downloadLink.download = "recorded_audio.wav";
                    downloadLink.click();

                    // Pass the audio as base64 to the Streamlit backend
                    const message = base64Audio;
                    window.parent.postMessage(message, "*");
                };
                reader.readAsDataURL(audioBlob);
            };

            recorder.start();
            window.setTimeout(() => recorder.stop(), 5000);  // Stop recording after 5 seconds
        }

        // Start recording when button is clicked
        startRecording();
    </script>
"""

# Display HTML content to record audio
html(html_code)

# Receiving audio from the browser
audio_base64 = st.text_input("Base64 Audio", "")

# If audio is available, process it
if audio_base64:
    handle_audio_upload(audio_base64)
