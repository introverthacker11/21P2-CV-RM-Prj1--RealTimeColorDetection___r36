import cv2
import numpy as np
import gradio as gr
import time

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

prev_time = time.time()  # For FPS calculation

# ---------------------------- Color Detection Function ----------------------------
def detect_color(frame, selected_color):
    global prev_time
    frame = cv2.resize(frame, (640, 480))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Compute mask
    color_values = colors_hsv[selected_color]
    if selected_color == "Red":
        lower1, upper1, lower2, upper2 = map(np.array, color_values)
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = mask1 + mask2
    else:
        lower, upper = map(np.array, color_values)
        mask = cv2.inRange(hsv, lower, upper)

    # Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Contours & bounding boxes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, selected_color, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    combined = np.hstack((frame, mask_rgb))

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(combined, f"FPS: {int(fps)}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return combined

# ---------------------------- Gradio Layout ----------------------------
selected_color_dropdown = gr.Dropdown(list(colors_hsv.keys()), label="Select Color")

# Project intro, developer info, and tech stack
project_intro = """
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
"""

developer_info = """
### 👨‍💻 Developer's Info
- **Hi, I'm Rayyan Ahmed**  
- **Google Certified AI Prompt Specialist**  
- **IBM Certified Advanced LLM FineTuner**  
- **Google Certified Soft Skill Professional**  
- **Hugging Face Certified: Fundamentals of LLMs**  
- **Expert in EDA, ML, RL, ANN, CNN, CV, RNN, NLP, LLMs**  
[💼 LinkedIn](https://www.linkedin.com/in/rayyan-ahmed-504725321/)
"""

tech_stack = """
### 🛠️ Tech Stack Used
**Python & Libraries:** Numpy, Pandas, Matplotlib, Seaborn  
**ML & AI:** Scikit-learn, TensorFlow, Keras, RL  
**Data Storage:** Pickle, CSV, JSON  
**Web & UI:** Gradio, PIL  
**Version Control & Deployment:** Git, Hugging Face Spaces
"""

# Background HTML
background_html = """
<style>
body {
    background-image: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)),
                      url("https://cdn.vectorstock.com/i/500p/87/89/two-dome-security-cameras-business-monitoring-vector-55888789.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: white;
    font-family: Arial, sans-serif;
}
h1 {
    color: #FFD700;
    text-align: center;
}
</style>
"""

# ---------------------------- Single Gradio Blocks ----------------------------
with gr.Blocks(title="⚡🤖 Real-Time Color Detection Web App") as demo:
    gr.HTML(background_html)
    gr.Markdown("**🎨 Detect and track selected colors in real-time using your webcam**  \n👨‍💻 Developed by **Rayyan Ahmed**")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(project_intro)
            gr.Markdown(developer_info)
            gr.Markdown(tech_stack)
        with gr.Column(scale=2):
            webcam_input = gr.Image(source="webcam", tool="editor", type="numpy")
            output_image = gr.Image(type="numpy")
            webcam_input.change(detect_color, inputs=[webcam_input, selected_color_dropdown], outputs=output_image)
            selected_color_dropdown.change(detect_color, inputs=[webcam_input, selected_color_dropdown], outputs=output_image)

demo.launch()
