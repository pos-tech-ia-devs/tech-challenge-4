# Tech Challenge 4

## Requirements
- Python 3.8-3.10

## How to Run

```
Be sure that the video named input_video.mp4 is on the root of the folder
```

First, you must create a virtual environment by running the following command:

```bash
py -m venv venv
```

> **Note**: You may need to call Python as `py` or `python` depending on your system.

To activate the virtual environment, use the following command:

```bash
source venv/bin/activate
```

If you're on Windows, use this command instead:

```bash
venv\Scripts\activate
```

Next, install the required packages by running:

```bash
pip install -r requirements.txt
```

Finally, you can run the application with the following command:

```bash
py detect_expression_video.py
```

## Results

The code will analyze and display the following statistics from the video:

1. **Frames Analyzed**: The total number of frames processed in the video.
2. **Anomalies Detected**: The count of anomalous movements identified during the analysis.
3. **Main Emotions**: A summary of the detected emotions and their respective counts.
4. **Main Activities**: A summary of detected pose-related actions (e.g., arm up, squat) and their respective counts.

### Best model `mtcnn` results [WIP]

Estatísticas do Vídeo:
1. **Frames Analisados**: 3326
2. **Anomalias Detectadas**: 144
3. **Emoções Principais**: {'neutral': 1018, 'sad': 473, 'fear': 444, 'angry': 248, 'happy': 516, 'surprise': 183}
4. **Atividades Principais**: {'head_tilt_right': 2278, 'head_tilt_left': 1610, 'arm_up': 624, 'both_arms_up': 240, 'squat': 468, 'leg_up': 324, 'jump': 199}

## Model Details for DeepFace detector backends

In `DeepFace.analyze`, the `detector_backend` parameter specifies the face detection model to be used. The available options for `detector_backend` are:

### `mtcnn`
- **Description**: Uses Multi-Task Cascaded Convolutional Networks (MTCNN) for face detection.
- **Performance**: Highly accurate, especially for detecting faces in challenging conditions (e.g., tilted or partially obscured faces). However, it is slower than `opencv` and `ssd`.
- **Usage**: Ideal for applications where accuracy is critical, and performance is less of a concern.

### `opencv`
- **Description**: Uses OpenCV's Haar Cascade Classifier for face detection.
- **Performance**: Fast and lightweight, but less accurate compared to other backends, especially in detecting faces in challenging conditions (e.g., tilted or partially obscured faces).
- **Usage**: Suitable for applications where speed is more critical than accuracy.

### `ssd`
- **Description**: Uses Single Shot Multibox Detector (SSD) for face detection.
- **Performance**: Balances speed and accuracy. It is more accurate than `opencv` but slower.
- **Usage**: Good for general-purpose face detection with moderate performance requirements.

### `dlib`
- **Description**: Uses Dlib's Histogram of Oriented Gradients (HOG) and Convolutional Neural Network (CNN) models for face detection.
- **Performance**: More accurate than `opencv` and `ssd`, but slower. The CNN model in Dlib is particularly accurate for detecting faces in various orientations.
- **Usage**: Suitable for applications requiring higher accuracy but can tolerate slower performance.

### `retinaface`
- **Description**: Uses RetinaFace, a state-of-the-art face detection model.
- **Performance**: Very accurate and robust, capable of detecting faces in challenging conditions. It is slower than `opencv` and `ssd` but faster than `mtcnn` in some cases.
- **Usage**: Suitable for high-accuracy applications with moderate performance requirements.

### `mediapipe`
- **Description**: Uses Google's MediaPipe framework for face detection.
- **Performance**: Fast and accurate, optimized for real-time applications.
- **Usage**: Ideal for real-time applications where both speed and accuracy are important.

### Summary of Performance and Usage
- **Fastest**: `opencv`, `mediapipe`
- **Most Accurate**: `mtcnn`, `retinaface`
- **Balanced**: `ssd`, `dlib`

The choice of `detector_backend` depends on the specific requirements of your application, such as speed, accuracy, and the complexity of the environment where faces are detected.
