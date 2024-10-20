# Vehicle Cut-In Detection - The Hawks

## Project Summary

This project employs YOLOv5 for real-time vehicle cut-in detection and collision warnings using a live video feed from a webcam. The model, integrated with PyTorch, is trained on the [IDD Dataset](https://idd.insaan.iiit.ac.in) to accurately identify vehicles and compute their distances from the camera. By calculating the Time-to-Collision (TTC) based on these distances, the system provides proactive collision alerts. Key technologies used include OpenCV for video processing, Streamlit for the user interface, and PyTorch for deep learning.

Key Features:
- **Real-Time Detection**: Uses YOLOv5 for efficient and accurate vehicle detection in live video feeds.
- **Distance and TTC Calculation**: Computes the distance of detected vehicles and TTC to predict possible collisions.
- **Visual and Audible Alerts**: Provides real-time visual annotations and audible warnings for imminent collisions.
- **Interactive UI**: Streamlit-based interface for monitoring the video feed and collision warnings dynamically.

The project demonstrates the integration of advanced deep learning techniques with practical safety applications, enhancing transportation safety and efficiency.