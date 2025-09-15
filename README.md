# 🎨 Real-Time Color Detection with OpenCV

This project is a fun beginner-friendly computer vision tool that detects the **dominant color** of objects you hold in front of your webcam. Instead of showing raw RGB values, it gives the actual **color name** (like red, blue, black, or pink). If nothing is detected, it smartly responds with *“You are not holding anything.”* Each detection is also logged into a `color_report.txt` file with timestamps.

---

## ✨ Features
- Real-time webcam-based color detection  
- Converts RGB values into human-readable color names  
- Ignores background, focusing only on the object you hold  
- Generates a timestamped report of detections  
- Beginner-friendly and easy to run  

---

## 🚀 Installation & Usage
```bash
git clone https://github.com/yourusername/color-detection-opencv.git
cd color-detection-opencv
pip install opencv-python numpy scikit-learn webcolors
python color_detection.py
```
