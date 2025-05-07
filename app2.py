import cv2
import streamlit as st
import numpy as np
import time
import random
from ultralytics import YOLO
from PIL import Image

# Streamlit Page Config
st.set_page_config(layout="wide")
st.title("Intelligent Traffic Management System")

# Upload Videos
st.subheader("Upload Traffic Videos")
video1 = st.file_uploader("Upload Video 1", type=["mp4", "avi", "mpeg"], key="video1")
video2 = st.file_uploader("Upload Video 2", type=["mp4", "avi", "mpeg"], key="video2")
video3 = st.file_uploader("Upload Video 3", type=["mp4", "avi", "mpeg"], key="video3")
video4 = st.file_uploader("Upload Video 4", type=["mp4", "avi", "mpeg"], key="video4")

# Load YOLO Model
model = YOLO(r"D:\Vishnu Clg\Project\C8\runs\detect\train12\weights\best.pt")

# Define Ambulance Class Index
AMBULANCE_CLASS = 0

# UI Layout (Grid Layout for 4 Videos)
cols = st.columns(4)
image_placeholders = [col.empty() for col in cols]
status_placeholders = [col.empty() for col in cols]

def process_videos(video_paths):
    caps = [cv2.VideoCapture(video) for video in video_paths]
    
    signal_states = [False] * 4
    signal_states[0] = True  # Initially, first signal is green
    
    # Display all videos at the start
    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if ret:
            frame_resized = cv2.resize(frame, (320, 240))
            img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
            image_placeholders[i].image(img, use_container_width=True)
            status_placeholders[i].markdown("### 🟢 Green" if i == 0 else "### 🔴 Red")
    
    while all(cap.isOpened() for cap in caps):
        for i in range(4):
            if not signal_states[i]:
                continue

            # Capture frame at start of green signal to determine vehicle count
            ret, frame = caps[i].read()
            if not ret:
                break
            
            frame_resized = cv2.resize(frame, (320, 240))
            results = model(frame_resized)
            
            boxes = results[0].boxes
            vehicle_count = len(boxes)
            ambulance_detected = any(int(box.cls) == AMBULANCE_CLASS for box in boxes)
            
            switch_duration = random.randint(20, 25) if vehicle_count > 8 else random.randint(10, 15)
            
            for remaining_time in range(switch_duration, 0, -1):
                ret, frame = caps[i].read()
                if not ret:
                    break
                
                frame_resized = cv2.resize(frame, (320, 240))
                results = model(frame_resized)
                
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls)
                    color = (0, 255, 0) if cls != AMBULANCE_CLASS else (0, 0, 255)
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
                
                img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
                image_placeholders[i].image(img, use_container_width=True)
                status_placeholders[i].markdown(f"### 🟢 Green - {remaining_time} sec | Vehicles: {vehicle_count} | Ambulance: {ambulance_detected}")
                time.sleep(1)
            
            for yellow_time in range(3, 0, -1):
                ret, frame = caps[i].read()
                if not ret:
                    break
                
                frame_resized = cv2.resize(frame, (320, 240))
                results = model(frame_resized)
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls)
                    color = (0, 255, 0) if cls != AMBULANCE_CLASS else (0, 0, 255)
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
                
                img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
                image_placeholders[i].image(img, use_container_width=True)
                status_placeholders[i].markdown(f"### 🟡 Yellow - {yellow_time} sec")
                time.sleep(1)
                
                if yellow_time == 1:
                    next_index = (i + 1) % 4
                    signal_states[next_index] = True
                    status_placeholders[next_index].markdown("### 🟢 Green")
            
            status_placeholders[i].markdown("### 🔴 Red")
            signal_states[i] = False
            time.sleep(random.randint(3, 6))
    
    for cap in caps:
        cap.release()

if st.button("Start Processing"):
    if all([video1, video2, video3, video4]):
        video_files = ["video1.mp4", "video2.mp4", "video3.mp4", "video4.mp4"]
        for i, video in enumerate([video1, video2, video3, video4]):
            with open(video_files[i], "wb") as f:
                f.write(video.read())
        st.write("Processing videos...")
        process_videos(video_files)
