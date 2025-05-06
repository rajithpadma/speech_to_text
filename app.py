import streamlit as st
import whisper
import tempfile
import os

# Set page title and icon
st.set_page_config(page_title="Whisper Transcriber", page_icon="🎙️")
st.title("🎙️ Whisper Speech-to-Text Transcriber")

st.markdown(
    "Upload a **.wav**, **.mp3**, or **.m4a** audio file. "
    "This app uses OpenAI's Whisper model to convert speech to text."
)

# Load the Whisper model once (cached)
@st.cache_resource
def load_model():
    return whisper.load_model("base")  # Try "small", "medium", or "large" for better accuracy

model = load_model()

# File uploader
uploaded_file = st.file_uploader("📂 Choose an audio file", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary file with correct extension
    file_ext = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Play the audio file in the app
    st.audio(tmp_path, format='audio/wav')

    st.info("⏳ Transcribing... please wait.")
    result = model.transcribe(tmp_path)

    # Show the transcription result
    st.subheader("📝 Transcription:")
    st.write(result["text"])

    # Provide a download button for the transcription
    st.download_button(
        label="💾 Download Transcription as .txt",
        data=result["text"],
        file_name="transcription.txt",
        mime="text/plain"
    )

    # Clean up the temp file
    os.remove(tmp_path)
