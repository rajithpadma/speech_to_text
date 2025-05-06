!pip install streamlit_webrtc
import streamlit_webrtc
import streamlit as st
import whisper
import tempfile
import os
import numpy as np
import av

try:
    from streamlit_webrtc import (
        webrtc_streamer,
        WebRtcMode,
        ClientSettings
    )
except ImportError as e:
    st.error("❌ Missing required package: `streamlit-webrtc`. Install it with `pip install streamlit-webrtc`.")
    raise e

# Set page config
st.set_page_config(page_title="Whisper Speech-to-Text", page_icon="🎙️")
st.title("🎙️ Whisper Speech-to-Text App")

st.markdown("Choose a mode to transcribe speech using OpenAI's Whisper model.")

# Load Whisper model once
@st.cache_resource
def load_model():
    return whisper.load_model("base")  # Use "small", "medium", "large" for more accuracy

model = load_model()

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
    class AudioProcessor:
        def __init__(self):
            self.frames = []

        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            self.frames.append(frame.to_ndarray())
            return frame

    ctx = webrtc_streamer(
        key="speech",
        mode=WebRtcMode.SENDONLY,
        in_audio=True,
        client_settings=ClientSettings(
            media_stream_constraints={"audio": True, "video": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        ),
        audio_processor_factory=AudioProcessor,
        async_processing=True,
    )

    if ctx.state.playing:
        st.warning("🎙️ Recording... Speak now.")
    elif ctx.audio_processor and ctx.audio_processor.frames:
        st.success("✅ Finished recording. Transcribing...")

        audio = np.concatenate(ctx.audio_processor.frames, axis=1).flatten().astype(np.int16).tobytes()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            import wave
            with wave.open(f, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio)
            temp_path = f.name

        try:
            result = model.transcribe(temp_path)
            st.subheader("📝 Transcription:")
            st.write(result["text"])
            st.download_button("💾 Download Transcript", result["text"], "transcription.txt")
        except Exception as e:
            st.error(f"❌ Transcription failed: {e}")
        finally:
            os.remove(temp_path)
