import streamlit as st
import tempfile
import whisper
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av
import numpy as np
import wave
import os

st.title("🎙️ Voice Recorder + Whisper Transcription")

# Whisper model
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# Temporary WAV file writer
def save_wav(audio, filename, sample_rate):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)

# Audio processor
class AudioProcessor:
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        self.frames.append(audio)
        return frame

st.info("Click below to start recording your voice.")

ctx = webrtc_streamer(
    key="speech",
    mode=WebRtcMode.SENDONLY,
    in_audio=True,
    client_settings=ClientSettings(
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    ),
    audio_receiver_size=1024,
    audio_processor_factory=AudioProcessor,
)

if ctx.state.playing:
    st.warning("🔴 Recording... Speak into the mic.")
else:
    if ctx.audio_receiver:
        # Access collected audio
        audio_processor = ctx.audio_processor
        if audio_processor and audio_processor.frames:
            audio_data = np.concatenate(audio_processor.frames, axis=1).flatten().astype(np.int16).tobytes()

            # Save to WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                save_wav(audio_data, f.name, sample_rate=ctx.audio_receiver.get_audio_frame_rate())
                audio_file_path = f.name

            st.success("✅ Audio recorded. Transcribing...")
            result = model.transcribe(audio_file_path)
            st.subheader("📝 Transcription:")
            st.write(result["text"])
            st.audio(audio_file_path, format="audio/wav")

            # Clean up
            os.remove(audio_file_path)
