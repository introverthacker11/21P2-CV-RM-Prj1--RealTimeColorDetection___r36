import cv2
import numpy as np
import streamlit as st
import time

st.title("⚡🤖 Real-Time Color Detection Web App")
st.markdown("**🎨 Detect and track selected colors in real-time using your webcam**  \n👨‍💻 Developed by **Rayyan Ahmed**")


# ---------------------------- Set background ----------------------------
st.markdown("""
<style>
.stApp {
    background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)),
                      url("https://cdn.vectorstock.com/i/500p/87/89/two-dome-security-cameras-business-monitoring-vector-55888789.jpg");
    background-size: 100%;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: white;
}
h1 { color: #FFD700; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ---------------------------- Sidebar ----------------------------
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: rgba(0, 0, 70, 0.45);
    color: white;
}
[data-testid="stSidebar"] h1, h2, h3 { color: #00171F; }
::-webkit-scrollbar-thumb { background: #00cfff; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar.expander("📌 Project Intro"):
    st.markdown("""
    ### 🎯 Project Goal
    - Detect selected colors in real-time from your webcam.  
    - Highlight detected objects with bounding boxes.  
    - Display the mask for visualization and debugging.  

    ### 🖼️ Features
    - Real-time color detection for **10 predefined colors**  
    - **Bounding boxes** and **color labels** for detected objects  
    - **Mask view** to see detected areas  
    - **FPS display** to monitor performance  

    ### ⚡ Use Cases
    - Educational purposes: Learn color detection and computer vision.  
    - Robotics: Object tracking based on color.  
    - DIY projects: Color-based sorting or interactive installations.  
    - Image & video processing experiments.  

    ### 🛠️ How It Works
    1. Capture video from webcam.  
    2. Convert frames to HSV color space.  
    3. Apply color mask based on user-selected color.  
    4. Clean mask using morphological operations.  
    5. Detect contours and draw bounding boxes around objects.  
    6. Display original frame with boxes alongside the mask.
    """)

# Developer's intro
with st.sidebar.expander("👨‍💻 Developer's Intro"):
    st.markdown("- **Hi, I'm Rayyan Ahmed**")
    st.markdown("- **Google Certified AI Prompt Specialist**")
    st.markdown("- **IBM Certified Advanced LLM FineTuner**")
    st.markdown("- **Google Certified Soft Skill Professional**")
    st.markdown("- **Hugging Face Certified: Fundamentals of LLMs**")
    st.markdown("- **Expert in EDA, ML, RL, ANN, CNN, CV, RNN, NLP, LLMs**")
    st.markdown("[💼 Visit LinkedIn](https://www.linkedin.com/in/rayyan-ahmed-504725321/)")

# Tech Stack
with st.sidebar.expander("🛠️ Tech Stack Used"):
    st.markdown("""
    ### 🐍 Python & Libraries
    - **Numpy** – Array & numerical computations  
    - **Pandas** – Data manipulation & analysis  
    - **Matplotlib & Seaborn** – Data visualization  

    ### 🤖 Machine Learning & AI
    - **Scikit-learn** – ML algorithms & preprocessing  
    - **TensorFlow & Keras** – Deep learning & neural networks  
    - **Reinforcement Learning (RL)** – Custom AI experiments  

    ### 💾 Data Storage & Serialization
    - **Pickle** – Save & load models  
    - **CSV / JSON** – Dataset handling  

    ### 🌐 Web App & UI
    - **Streamlit** – Interactive web apps  
    - **PIL (Pillow)** – Image processing  

    ### ⚙️ Version Control & Deployment
    - **Git** – Source code management  
    - **Streamlit Cloud** – Deployment & sharing
    """)


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

# ---------------------------- Streamlit Color Selection ----------------------------
selected_color = st.selectbox("Select Color to Detect", list(colors_hsv.keys()))

# Start webcam
cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to grab frame")
        break

    frame = cv2.resize(frame, (640, 480))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

    # ------------------ Compute mask for current frame ------------------
    color_values = colors_hsv[selected_color]
    if selected_color == "Red":
        lower1, upper1, lower2, upper2 = map(np.array, color_values)
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = mask1 + mask2
    else:
        lower, upper = map(np.array, color_values)
        mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame_rgb, selected_color, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    combined = np.hstack((frame_rgb, mask_rgb))

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(combined, f"FPS: {int(fps)}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Display in Streamlit
    frame_placeholder.image(combined, channels="RGB")

    time.sleep(0.01)

cap.release()
