import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="⚡🤖 Real-Time Color Detection", layout="wide")

st.title("⚡🤖 Real-Time Color Detection Web App")
st.markdown("**🎨 Detect and track selected colors in real-time using your webcam**  \n👨‍💻 Developed by **Rayyan Ahmed**")

# ---------------------------- Sidebar ----------------------------
selected_color = st.selectbox("Select Color to Detect", 
                              ["Yellow", "Red", "Green", "Blue", "Orange", "Purple", "Pink", "Cyan", "Brown", "White"])

# ---------------------------- HSV Color Ranges ----------------------------
colors_hsv = {
    "Yellow": ([20, 100, 100], [30, 255, 255]),
    "Red": ([0, 120, 70], [10, 255, 255], [170, 120, 70], [180, 255, 255]),
    "Green": ([36, 50, 70], [89, 255, 255]),
    "Blue": ([94, 80, 2], [126, 255, 255]),
    "Orange": ([10, 100, 20], [25, 255, 255]),
    "Purple": ([129, 50, 70], [158, 255, 255]),
    "Pink": ([160, 50, 70], [170, 255, 255]),
    "Cyan": ([80, 100, 100], [100, 255, 255]),
    "Brown": ([10, 100, 20], [20, 255, 200]),
    "White": ([0, 0, 200], [180, 25, 255])
}

# ---------------------------- Transformer Class ----------------------------
class ColorDetector(VideoTransformerBase):
    def __init__(self):
        self.selected_color = selected_color

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Generate mask
        if self.selected_color == "Red":
            lower1, upper1, lower2, upper2 = map(np.array, colors_hsv["Red"])
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = mask1 + mask2
        else:
            lower, upper = map(np.array, colors_hsv[self.selected_color])
            mask = cv2.inRange(hsv, lower, upper)

        # Morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Draw contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(img, self.selected_color, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---------------------------- Start Webcam ----------------------------
webrtc_streamer(
    key="color-detection",
    video_transformer_factory=ColorDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True
)
