import cv2
import numpy as np
import streamlit as st
import torch
import time
from playsound import playsound
import threading
import os

@st.cache(allow_output_mutation=True)
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s')
vehicle_classes = [2, 3, 5, 7]  

def distance_to_camera(known_width, focal_length, pixel_width):
    return (known_width * focal_length) / pixel_width
def calculate_ttc(distance, relative_speed):
    if relative_speed <= 0:
        return float('inf')
    return distance / relative_speed

def play_warning_sound():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sound_file = os.path.join(current_dir, '/Users/balajia/Desktop/Intel project/Alarm Buzzer Ringtone Download - MobCup.Com.Co.mp3')
    
    if os.path.isfile(sound_file):
        try:
            print(f"Playing warning sound from {sound_file}")
            playsound(sound_file)
        except Exception as e:
            print(f"Error playing sound: {e}")
    else:
        print(f"Warning sound file not found: {sound_file}")

def detect_and_draw(frame, model, focal_length, known_vehicle_width, ego_speed, prev_detections):
    results = model(frame, size=640) 
    detections = results.xyxy[0].detach().cpu().numpy()
    current_detections = []
    for box in detections:
        x1, y1, x2, y2, conf, cls = box
        if int(cls) in vehicle_classes:
            pixel_width = x2 - x1
            distance = distance_to_camera(known_vehicle_width, focal_length, pixel_width)
            detected_vehicle_speed = 20 
            relative_speed = ego_speed - detected_vehicle_speed
            ttc = calculate_ttc(distance, relative_speed)
            current_detections.append((x1, y1, x2, y2, model.names[int(cls)], conf, distance, ttc))

    for detection in current_detections:
        x1, y1, x2, y2, cls_name, conf, distance, ttc = detection
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = f'{cls_name} {conf:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f'Distance: {distance:.2f} meters', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f'TTC: {ttc:.2f} seconds', (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    nearest_vehicle = min(current_detections, key=lambda x: x[6], default=None)
    if nearest_vehicle:
        x1, y1, x2, y2, cls_name, conf, distance, ttc = nearest_vehicle
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = f'{cls_name} {conf:.2f}'

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, f'Distance: {distance:.2f} meters', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, f'TTC: {ttc:.2f} seconds', (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if ttc < 0.7:
            cv2.putText(frame, 'WARNING: COLLISION IMMINENT', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            threading.Thread(target=play_warning_sound).start()

    prev_detections.clear()
    prev_detections.extend(current_detections)

    return frame

def main():
    st.title("Live Vehicle Detection and Collision Warning from Webcam")
    model = load_model()
    start_detection = st.button("Start Vehicle Detection")

    if start_detection:
        cap = cv2.VideoCapture(0)
        focal_length = 1000 
        known_vehicle_width = 1.8 
        ego_speed = 30 
        prev_detections = []
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = detect_and_draw(frame, model, focal_length, known_vehicle_width, ego_speed, prev_detections)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB", use_column_width=True)
                time.sleep(0.1)
            else:
                break

        cap.release()

if __name__ == "__main__":
    main()
