
import av
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(
    page_title="RASTHEX",
    page_icon="🚧",
    layout="wide"
)

st.title("🚧 RASTHEX")
st.subheader("AI-Powered Road Intelligence & Civic Reporting System")
st.write("Live pothole detection using a camera.")
st.divider()


@st.cache_resource
def load_model():
    return YOLO("../models/best.pt")


model = load_model()


class PotholeDetector(VideoProcessorBase):

    def __init__(self):
        self.frame_width = 0
        self.frame_height = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        self.frame_height, self.frame_width = img.shape[:2]

        results = model.track(
            img,
            conf=0.7,
            persist=True,
            verbose=False
        )

        annotated_frame = results[0].plot(
            line_width=3,
            font_size=1.0
        )

        return av.VideoFrame.from_ndarray(
            annotated_frame,
            format="bgr24"
        )


st.subheader("Live Pothole Detection")

ctx = webrtc_streamer(
    key="rasthex-camera",
    video_processor_factory=PotholeDetector,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1280},
            "height": {"ideal": 720},
            "frameRate": {"ideal": 30}
        },
        "audio": False
    },
    async_processing=True
)

st.subheader("Camera Information")

if ctx.video_processor:
    width = ctx.video_processor.frame_width
    height = ctx.video_processor.frame_height

    if width > 0 and height > 0:
        st.write(f"Actual camera resolution: {width} x {height}")
    else:
        st.info("Waiting for camera frames...")
else:
    st.info("Click START to activate the camera.")

st.divider()
