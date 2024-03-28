# Vehicle Speed Estimation using Computer Vision

## Overview
This project demonstrates a complete system for estimating the speed of vehicles using computer vision techniques. By integrating object detection, multi-object tracking, and perspective transformation, the system calculates the speed of vehicles in real-time from video footage.
Check out the [video demonstration here](https://drive.google.com/drive/folders/1arEAoWKoFXuqoXD7FPxJ1ydpRjwgQ3hg?usp=sharing)

## Features
- Utilizes advanced object detection models for accurate vehicle detection.
- Employs BYTETrack for effective multi-object tracking.
- Applies perspective transformation to map image coordinates to real-world measurements for speed calculation.

## Methodology
### Object Detection
The system begins with object detection, iterating over video frames and applying a pre-trained YOLO model for vehicle detection. This process is facilitated by the Supervision library for video processing and annotation.

### Multi-Object Tracking
With vehicles detected, BYTETrack is used to track each vehicle across frames. This is crucial for calculating speed, as it allows the system to measure the distance each vehicle travels over time.

### Perspective Distortion Correction
A challenge in measuring real-world speed from video is perspective distortion. The project addresses this by applying a perspective transformation, converting image coordinates into actual coordinates on the road.

## Challenges & Solutions
- **Perspective Distortion**: Solved using OpenCV's `getPerspectiveTransform` function to create a transformation matrix, enabling accurate distance measurements.
- **Speed Calculation**: To mitigate detection flickering effects and improve accuracy, speed calculations are averaged over one-second intervals, providing stable and realistic speed estimations.

## Technical Highlights
- **Homography**: Mathematically determines a transformation matrix for perspective correction.
- **Speed Estimation Logic**: Combines tracking data with frame rate information to calculate vehicle speeds.

## Hidden Complexities
- **Occlusions**: Handled through robust object tracking, ensuring accurate speed estimation even when vehicles are partially obscured.
- **Reference Point Selection**: Utilizes the bottom center of bounding boxes as a consistent reference point for speed calculation.

## Conclusion
This project illustrates the intricacies of estimating vehicle speed using computer vision. By overcoming challenges such as perspective distortion and object tracking, it offers a framework that can be adapted and extended for various applications.

## Future Work
Exploring additional complexities such as varying weather conditions, road slopes, and different traffic densities can further enhance the system's robustness and accuracy.

## Acknowledgments
- **Object Detection Models**: Special thanks to the developers of YOLO and the Roboflow model repository.
- **Tracking Algorithm**: Gratitude to the creators of BYTETrack and the Supervision
