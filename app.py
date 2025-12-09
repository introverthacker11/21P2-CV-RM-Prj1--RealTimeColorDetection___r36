import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="⚡🤖 Real-Time Color Detection", layout="wide")
st.title("⚡🤖 Real-Time Color Detection Web App")
st.markdown("**🎨 Detect and track selected colors in real-time using your webcam**  \n👨‍💻 Developed by **Rayyan Ahmed**")

# ---------------- Sidebar ----------------
with st.sidebar.expander("📌 Project Intro"):
    st.markdown("""
    - Detect selected colors in real-time from your webcam  
    - Highlight detected objects with bounding boxes  
    - Display FPS for performance monitoring
    """)

with st.sidebar.expander("👨‍💻 Developer's Intro"):
    st.markdown("""
    - **Rayyan Ahmed**  
    - Google Certified AI Prompt Specialist  
    - IBM Certified Advanced LLM FineTuner  
    - Expert in CV, ML, LLMs  
    [💼 LinkedIn](https://www.linkedin.com/in/rayyan-ahmed-504725321/)
    """)

# ---------------- Color Selection ----------------
selected_color = st.selectbox("Select Color to Detect", 
                              ["Yellow", "Red", "Green", "Blue", "Orange", "Purple", "Pink", "Cyan", "Brown", "White"])

# ---------------- HSV Ranges ----------------
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

# ---------------- Video Transformer ----------------
class ColorDetector(VideoTransformerBase):
    def __init__(self):
        self.color_name = selected_color
        self.prev_time = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Red has two ranges
        if self.color_name == "Red":
            lower1, upper1, lower2, upper2 = map(np.array, colors_hsv["Red"])
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = mask1 + mask2
        else:
            lower, upper = map(np.array, colors_hsv[self.color_name])
            mask = cv2.inRange(hsv, lower, upper)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours and draw bounding boxes
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(img, self.color_name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # FPS overlay
        import time
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time) if self.prev_time != 0 else 0
        self.prev_time = curr_time
        cv2.putText(img, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return img

# ---------------- WebRTC Streamer ----------------
webrtc_streamer(
    key="color-detection",
    video_transformer_factory=ColorDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
