# Tech Challenge 4

## Requirements
- Python 3.8-3.10

## How to Run

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

### Model `mtcnn` results [WIP]
```
DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, align=True,detector_backend=DETECTOR_BACKEND_MTCNN)
```
Video Statistics:
1. **Frames Analyzed**:  3326
2. **Anomalies Detected**: 142
3. **Main Emotions**: {'neutral': 1222, 'sad': 884, 'fear': 580, 'angry': 310, 'happy': 856, 'surprise': 188}
4. **Main Activities**: {'head_tilt_right': 2134, 'head_tilt_left': 1590, 'squat': 374, 'leg_up': 231, 'arm_up': 507, 'both_arms_up': 237, 'jump': 148}

### Model `mediapipe` results [DONE]
```
DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, align=True,detector_backend=DETECTOR_BACKEND_MEDIAPIPE)
```
Video Statistics:
1. **Frames Analyzed**:  3326
2. **Anomalies Detected**: 152
3. **Main Emotions**: {'angry': 239, 'neutral': 737, 'disgust': 6, 'fear': 584, 'sad': 859, 'surprise': 77, 'happy': 824}
4. **Main Activities**: {'head_tilt_right': 2151, 'head_tilt_left': 1683, 'arm_up': 580, 'leg_up': 279, 'squat': 316, 'both_arms_up': 239, 'jump': 147}

### Model `retinaface` results [CANCELLED]
```
DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, align=True,detector_backend=DETECTOR_BACKEND_RETINAFACE)
```

### Model `opencv` results [WIP]
```
DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, align=True,detector_backend=DETECTOR_BACKEND_OPENCV)
```

### Model `dlib` results [WIP]
```
DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, align=True,detector_backend=DETECTOR_BACKEND_DLIB)
```

### Model `ssd` results [WIP]
```
DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, align=True,detector_backend=DETECTOR_BACKEND_SSD)
```

### Model `mediapipe` results [WIP]
```
DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, align=True,detector_backend=DETECTOR_BACKEND_MEDIAPIPE)
```


## Model Details for DeepFace detector backends

In `DeepFace.analyze`, the `detector_backend` parameter specifies the face detection model to be used. The available options for `detector_backend` are:

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

### `mtcnn`
- **Description**: Uses Multi-Task Cascaded Convolutional Networks (MTCNN) for face detection.
- **Performance**: Highly accurate, especially for detecting faces in challenging conditions (e.g., tilted or partially obscured faces). However, it is slower than `opencv` and `ssd`.
- **Usage**: Ideal for applications where accuracy is critical, and performance is less of a concern.

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
