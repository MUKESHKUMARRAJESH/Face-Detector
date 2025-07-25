import streamlit as st
import cv2
import numpy as np
from datetime import datetime

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Function to detect faces and draw rectangles
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return frame, len(faces)

# Title
st.title("🧠 Real-Time Face Detection App")
st.markdown("This app uses **OpenCV** and **Streamlit** to detect faces from your webcam.")

# Buttons and checkbox
start = st.checkbox("Start Camera")
capture = st.button("📸 Capture Frame")
show_count = st.checkbox("Show face count")

# Streamlit camera video capture
if start:
    # Open webcam (0 is the default webcam)
    cap = cv2.VideoCapture(0)

    # Display video stream
    frame_placeholder = st.empty()
    count_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Camera not accessible.")
            break

        # Detect faces
        frame, face_count = detect_faces(frame)

        # Show face count if enabled
        if show_count:
            count_placeholder.markdown(f"### 👥 Faces detected: {face_count}")
        else:
            count_placeholder.empty()

        # Convert to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Save frame if capture is clicked
        if capture:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captured_face_{timestamp}.png"
            cv2.imwrite(filename, frame)
            st.success(f"Frame captured and saved as {filename}")
            capture = False  # Reset button

    cap.release()
else:
    st.info("☝️ Tick the checkbox to start webcam.")

