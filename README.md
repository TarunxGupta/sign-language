# HandsFree: Real-Time Hand Gesture Recognition System

HandsFree is an innovative real-time hand gesture recognition system leveraging deep learning and computer vision technologies. Designed for touchless interaction, the project uses standard webcams, MediaPipe for hand landmark detection, and TensorFlow/Keras for gesture classification. The system is cost-effective, robust, and adaptable for various applications like accessibility tools, smart devices, and more.

## Table of Contents

- [Introduction](#introduction)  
- [Features](#features)  
- [Technologies Used](#technologies-used)  
- [Setup and Installation](#setup-and-installation)  
- [How It Works](#how-it-works)  
- [Applications](#applications)
- [Contributing](#contributing)  
- [License](#license)

## Introduction

HandsFree bridges the gap between humans and technology by offering a natural way to interact using hand gestures. The system recognizes gestures like "Peace," "Close," and "Ok" in real time, providing accurate and intuitive touchless control. Key highlights include:

- **Data Collection and Augmentation:** Gesture images collected via a webcam, processed with MediaPipe.  
- **Deep Learning Model:** A lightweight neural network trained on normalized hand landmarks for high accuracy.  
- **Real-Time Inference:** Optimized for low-latency performance.

## Features

- **Real-Time Gesture Recognition**: Classifies gestures with high accuracy.  
- **Robust Landmark Detection**: Uses MediaPipe for precise hand position extraction.  
- **Environment Adaptability**: Works well across varying lighting, backgrounds, and orientations.  
- **Customizability**: Easily scalable to include new gestures.  
- **Cross-Platform Compatibility**: Runs on Windows, macOS, and Linux.

## Technologies Used

- **Python**: Core programming language.  
- **OpenCV**: For video capture and image processing.  
- **MediaPipe**: For hand detection and landmark extraction.  
- **TensorFlow/Keras**: To build and train the deep learning model.  
- **NumPy**: For numerical operations and dataset handling.

## Setup and Installation

1. **Clone the Repository**:  
   ```bash
   git clone https://github.com/TarunxGupta/HandsFree.git

2. **Install Dependencies**:
   ```bash
   pip install requirements.txt

3. **Run the project**

## How It Works

1. **Input Capture**: Webcam captures real-time video frames.
2. **Hand Detection**: MediaPipe detects hand(s) and extracts 21 landmarks.
3. **Feature Vector Creation**: Landmarks normalized for consistency.
4. **Gesture Classification**: Preprocessed data fed to the trained neural network.
5. **Output**: Displays recognized gesture with visual feedback.

## Applications

1. **Human-Computer Interaction**: Control devices using gestures.
2. **Accessibility Tools**: Assistive technologies for individuals with disabilities.
3. **Smart Home Control**: Manage appliances via intuitive gestures.
4. **Gaming & AR/VR**: Immersive gesture-based interactions.

## Contributing

Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request with your changes.

## License

This project is licensed under the MIT License.
