# Driving Coach
CSE 60535 Computer Vision Course Project, University of Notre Dame

## Goal
Create an application that analyzes dashcam videos and identifies and tracks objects relevant to the driver, like cars, lanes, signs, etc. The software will be able to detect/track multiple objects at a time, on a stream of video.

## Dataset
Datasets to be used for the project
- [KITTI Multi-object Tracking](https://www.cvlibs.net/datasets/kitti/eval_tracking.php) (images) for vihecle tracking
- [KITTI Road](https://www.cvlibs.net/datasets/kitti/eval_road.php) for lane detection

### Differences between Training and Validation Datasets

 - KITTI Multi-object Tracking
   - Training: Images + Object Information (bounding box, location, size, class, etc.)
   - Validation: Images Only

 - KIITI Road
   - Training: Images + Lane Segmentations (two types: road - the road area, i.e, the composition of all lanes, and lane - the ego-lane, i.e., the lane the vehicle is currently driving on)
   - Validation: Images Only

### Dataset Details

 - KITTI Multi-object Tracking
   - 21 training sequences and 29 test sequences
   - Resolution: 1242 $\times$ 375
   - Labeled 8 different classes: DontCare, Car, Van, Pedestrian, Truck, Cyclist, Person sitting, Tram (focusing on Car and Pedestrian only)
   - Miscs: The labeling process was done in two steps: First, annotators labeled 3D bounding boxes as tracklets in point clouds. Since a single 3D bounding box often poorly fits pedestrian tracklets, the left/right boundaries of each object were labeled using Mechanical Turk. Additionally, the object's occlusion state was labeled, and truncation was computed by backprojecting a car/pedestrian model into the image plane.

 - KITTI Road
   - 289 training and 290 test images
   - Resolution: 1242 $\times$ 375
   - Categories:
     - uu - urban unmarked (98/100)
     - um - urban marked (95/96)
     - umm - urban multiple marked lanes (96/94)
     - urban - combination of the three above
    
### Setup Details
 - 1 Inertial Navigation System (GPS/IMU): OXTS RT 3003
 - 1 Laserscanner: Velodyne HDL-64E
 - 2 Grayscale cameras, 1.4 Megapixels: Point Grey Flea 2 (FL2-14S3M-C)
 - 2 Color cameras, 1.4 Megapixels: Point Grey Flea 2 (FL2-14S3C-C)
 - 4 Varifocal lenses, 4-8 mm: Edmund Optics NT59-917
 - Laser scanner spinning speed: 10 fps (capturing approximately 100k points per cycle)
 - Vertical resolution of the laser scanner: 64

## Method
Possible solutions may be found from computer vision tasks related to image segmentation, object detection and object tracking.

### Object Detection
1. **YOLO (You Only Look Once)**
   - **Description**: A real-time object detection system that can detect multiple objects in an image with high speed and accuracy.
   - **Versions**: YOLOv3, YOLOv4, YOLOv5, YOLOv7, YOLOv8.
   - **Use Case**: Suitable for real-time applications like dashcam video analysis.
   - **Resources**: [YOLOv5 GitHub](https://github.com/ultralytics/yolov5), [YOLOv4 Paper](https://arxiv.org/abs/2004.10934), [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)

2. **SSD (Single Shot MultiBox Detector)**
   - **Description**: A fast object detection algorithm that uses a single deep neural network to detect objects in images.
   - **Use Case**: Useful for applications requiring a balance between speed and accuracy.
   - **Resources**: [SSD Paper](https://arxiv.org/abs/1512.02325)

3. **Faster R-CNN**
   - **Description**: A two-stage object detection system with high accuracy, often used for more precise detection tasks.
   - **Use Case**: Suitable for applications where accuracy is more critical than speed.
   - **Resources**: [Faster R-CNN Paper](https://arxiv.org/abs/1506.01497)

### Object Tracking
1. **SORT (Simple Online and Realtime Tracking)**
   - **Description**: A fast and simple tracking algorithm that uses the Hungarian algorithm and Kalman filters.
   - **Use Case**: Good for real-time tracking applications.
   - **Resources**: [SORT GitHub](https://github.com/abewley/sort)

2. **Deep SORT**
   - **Description**: An extension of SORT that integrates deep learning for re-identifying objects, improving tracking performance, especially in crowded scenes.
   - **Use Case**: Ideal for applications requiring robust tracking in complex environments.
   - **Resources**: [Deep SORT GitHub](https://github.com/nwojke/deep_sort)

3. **ByteTrack**
   - **Description**: A high-performance tracking algorithm designed to work well with various object detection models, achieving state-of-the-art results.
   - **Use Case**: Suitable for applications needing high accuracy and robustness.
   - **Resources**: [ByteTrack GitHub](https://github.com/ifzhang/ByteTrack)

### Lane Detection
1. **Hough Transform**
   - **Description**: A classical method for detecting lines in images, often used for simple lane detection tasks.
   - **Use Case**: Useful for straightforward scenarios with clear lane markings.
   - **Resources**: [Hough Transform Tutorial](https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html)

2. **Deep Learning-Based Methods**
   - **Description**: Methods like SCNN (Spatial CNN) specifically designed for lane detection in challenging conditions.
   - **Use Case**: Suitable for complex scenarios with occlusions and varying lighting conditions.
   - **Resources**: [SCNN Paper](https://arxiv.org/abs/1712.06080)

### Semantic Segmentation
1. **U-Net**
   - **Description**: A convolutional network designed for biomedical image segmentation but widely used for various segmentation tasks.
   - **Use Case**: Suitable for segmenting lanes, road signs, and other relevant features in dashcam footage.
   - **Resources**: [U-Net Paper](https://arxiv.org/abs/1505.04597)

2. **DeepLab**
   - **Description**: A deep learning model for semantic image segmentation, providing high accuracy and flexibility.
   - **Use Case**: Ideal for detailed segmentation tasks in complex scenes.
   - **Resources**: [DeepLab GitHub](https://github.com/tensorflow/models/tree/master/research/deeplab)

To ensure a correct and effective implementation, we will place priority to methods that are open-source and have widespread practical use, like YOLO, Deep Sort and Byte Track. We will implement based on their code released on GitHub, and innovatively integrate them with other methods and our ideas to further improve the effectiveness. 

Here is a good tutorial for reference: [Football Player Tracking](https://www.youtube.com/watch?v=QCG8QMhga9k).

## Implementation Workflow
1. **Preprocessing**:
   - Convert videos to frames.
   - Normalize and resize images.

2. **Model Training**:
   - Train YOLO to detect objects in each frame.
   - Train Deep SORT to track detected objects across frames.

3. **Post-Processing**:
   - Filter out false positives.
   - Apply non-maximum suppression to refine detections.

4. **Integration**:
   - Combine results from YOLO and Deep Sort.
   - Overlay detected objects and lanes on the original video frames.

5. **Evaluation**:
   - Use metrics like mAP (mean Average Precision) for object detection.
   - Use MOTA (Multiple Object Tracking Accuracy) for tracking performance.
   - Evaluate the model performance on the test set.
  
6. **Application Development**:
   - Build the frontend for users to interact, like uploading videos and see the tracking results, using HTML, CSS and JavaScript. Use React to improve the interaction and beautify the interface.
   - Build the backend for deploying our trained model to analyze the uploaded videos using Flask.
