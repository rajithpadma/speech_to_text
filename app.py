import streamlit as st
import whisper
import tempfile
import os
import numpy as np
import av

# Safe import for streamlit_webrtc
try:
    import streamlit_webrtc as webrtc
    WebRtcMode = webrtc.WebRtcMode
    ClientSettings = webrtc.ClientSettings
    webrtc_streamer = webrtc.webrtc_streamer
except ImportError:
    webrtc_streamer = None
    WebRtcMode = None
    ClientSettings = None

# Set page title
st.set_page_config(page_title="Whisper Speech-to-Text", page_icon="🎙️")
st.title("🎙️ Whisper Speech-to-Text App")
st.markdown("Choose a mode below to transcribe speech using OpenAI's Whisper model.")

# Load model
@st.cache_resource
def load_model():
    return whisper.load_model("base")  # You can change to "small", "medium", "large"

model = load_model()

# Mode selection
mode = st.radio("Select Mode:", ["📁 Upload Audio File", "🎤 Live Recording (Mic)"])

# --- Upload Mode ---
if mode == "📁 Upload Audio File":
    uploaded_file = st.file_uploader("Upload a .wav, .mp3, or .m4a file", type=["wav", "mp3", "m4a"])
    
    if uploaded_file is not None:
        file_ext = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        st.audio(tmp_path, format='audio/wav')
        st.info("⏳ Transcribing...")

        try:
            result = model.transcribe(tmp_path)
            st.subheader("📝 Transcription:")
            st.write(result["text"])

            st.download_button(
                label="💾 Download Transcription",
                data=result["text"],
                file_name="transcription.txt",
                mime="text/plain"
            )
        except Exception as e:
            st.error(f"❌ Transcription failed: {e}")
        finally:
            os.remove(tmp_path)

# --- Live Recording Mode ---
elif mode == "🎤 Live Recording (Mic)":
    if webrtc_streamer is None:
        st.error("❌ `streamlit-webrtc` is not installed or couldn't be imported. Install it to enable live mic recording.")
    else:
        class AudioProcessor:
            def __init__(self):
                self.frames = []

            def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
                audio = frame.to_ndarray()
                self.frames.append(audio)
                return frame

        ctx = webrtc_streamer(
            key="speech-to-text",
            mode=WebRtcMode.SENDONLY,
            in_audio=True,
            client_settings=ClientSettings(
                media_stream_constraints={"audio": True, "video": False},
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            audio_processor_factory=AudioProcessor,
            async_processing=True
        )

        if ctx.state.playing:
            st.warning("🔴 Recording... Speak into your microphone.")
        elif ctx.audio_processor:
            if ctx.audio_processor.frames:
                st.success("✅ Recording finished. Transcribing...")

                audio_data = np.concatenate(ctx.audio_processor.frames, axis=1).flatten().astype(np.int16).tobytes()
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    import wave
                    with wave.open(f, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        wf.writeframes(audio_data)
                    audio_path = f.name

                try:
                    result = model.transcribe(audio_path)
                    st.subheader("📝 Transcription:")
                    st.write(result["text"])

                    st.download_button(
                        label="💾 Download Transcription",
                        data=result["text"],
                        file_name="transcription.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"❌ Transcription failed: {e}")
                finally:
                    os.remove(audio_path)
