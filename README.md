# Real-Time Face Recognition & Emotion Tracker

A real-time computer vision pipeline that performs identity verification and emotion classification using a standard webcam. Built with OpenCV, `face_recognition`, and DeepFace, this project features custom facial alignment algorithms to significantly improve the accuracy of emotion prediction on tightly cropped tracking boxes.

## Features

* **Real-Time Identity Matching:** Uses dlib-based facial encodings to verify the target user against a reference image in real-time.
* **Optimized Emotion Classification:** Bypasses redundant Haar-cascade detections by utilizing existing facial landmarks. Custom affine transformations align the face to eye-level before passing it to the DeepFace CNN, preventing the "unaligned whole-crop" failure mode common in default implementations.
* **Temporal Smoothing:** Implements a rolling history buffer with majority-voting to eliminate single-frame misclassifications and stabilize the output label.
* **Performance Tuned:** Downscales the detection pass while preserving full-resolution encoding, maintaining high FPS without sacrificing identity matching accuracy.

## Installation

1. **Clone the repository:**
    git clone https://github.com/yourusername/face-identity-tracker.git
    cd face-identity-tracker

2. **Set up a virtual environment (Recommended):**
    python -m venv venv
    source venv/bin/activate

3. **Install dependencies:**
    pip install -r requirements.txt
    
   *Note: `face_recognition` requires `dlib`. Depending on your OS, you may need to install CMake and Visual Studio C++ Build Tools before compiling.*

## Usage

1. **Add a Reference Image:** Place a clear, well-lit frontal photo of your face inside the `assets/` folder and name it `reference_face.jpeg`.

2. **Run the Tracker:**
    python main.py

3. **Controls:**
   * Press **q** to quit the video stream.

## Configuration
You can easily adjust the performance and strictness of the tracker by editing the variables at the top of `main.py`:
* `MY_NAME` (default: "john doe"): The display name attached to the recognized face.
* `TOLERANCE` (default: 0.50): Lower numbers make face matching stricter.
* `DOWNSCALE` (default: 0.5): The scale factor for the detection pass.
* `EMOTION_EVERY` (default: 2): Run the DeepFace emotion model every Nth frame to save compute power.