import streamlit as st
import whisper
import tempfile
import os
import base64
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

# Function to save audio data to a .wav file
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

# Function to display HTML for recording audio
def record_audio_html():
    return """
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

# Mode selection: Upload or Record
mode = st.radio("Select Mode:", ["📁 Upload Audio File", "🎤 Record Audio"])

if mode == "📁 Upload Audio File":
    # Upload file mode
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

elif mode == "🎤 Record Audio":
    # Record audio mode using JavaScript
    html(record_audio_html())

    # Receiving audio from the browser
    audio_base64 = st.text_input("Base64 Audio", "")

    if audio_base64:
        handle_audio_upload(audio_base64)
