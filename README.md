A real-time emotion detection system using OpenCV in Python. Detects emotions such as Happy, Sad, Angry, Surprised, Tired, and Neutral without requiring DLIB or deep learning models.

Features

Real-time face and emotion detection using OpenCV

Advanced facial feature analysis: smile intensity, eye openness, brightness, contrast, face symmetry

Vibrant emotion dashboard with confidence bar

Works with just a webcam, no heavy dependencies

Temporal smoothing for stable emotion recognition

Installation

Clone the repository:

git clone <your-repo-link>


Navigate to the folder:

cd emotion-detector-python


Install dependencies:

pip install opencv-python numpy

Usage

Run the detector:

python <your-script-name>.py


Make sure your webcam is connected.

Press Q to quit the program.

How It Works

Detects faces, eyes, and smiles using Haar cascades.

Calculates facial features (smile intensity, eye openness, brightness, contrast, symmetry).

Uses a scoring system to estimate emotions.

Displays a real-time dashboard with emotion and confidence.
