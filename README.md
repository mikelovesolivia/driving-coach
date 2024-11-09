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

### Object Tracking
1. **Deep SORT**
   - **Description**: An extension of SORT that integrates deep learning for re-identifying objects, improving tracking performance, especially in crowded scenes.
   - **Use Case**: Ideal for applications requiring robust tracking in complex environments.
   - **Resources**: [Deep SORT GitHub](https://github.com/nwojke/deep_sort)

2. **Optical Flow and SIFT**


### Road Detection
1. **Hough Transform**
   - **Description**: A classical method for detecting lines in images, often used for simple lane detection tasks.
   - **Use Case**: Useful for straightforward scenarios with clear lane markings.
   - **Resources**: [Hough Transform Tutorial](https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html)

2. **Canny Edge Detection**
    
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


## Report

### First Update

#### Overview
In this first update, I preprocessed and implemented road detection on images from KITTI Road dataset. The following methods are employed:

- Preprocessing: RGB to gray, image cropping
- Gaussian Blurring: to reduce the noise and unimportant details;
- Canny Edge Detection: to detect the edges of the image, where the road is included;
- Hough Transform: to detect lines related to the road from the edges;
- Convex Hull: to identify the boundaries of detected lines and combine them to form the road segmentation;
- Intersection over Union (IoU): to evaluate the effectiveness of the detection result quantitatively.

#### Method
Here is a detailed description of the pipeline: 
1. A random KITTI Road image was selected and transformed to gray image. 
2. As a preprocessing step, Gaussian blurring was applied to remove the high-frequency noise incurred by varying lighting and shades and surrounding objects like trees, and reduce the harshness of edges so that only prominent edges will be detected.
3. Conducted Canny edge detection on the preprocessed image. Since roads often have distinct and clear boundaries, edge detection helps to find the sharp and clean edges that contribute to the finding of road boundaries. 
4. The edge image was cropped so that only the lower part (a polygon) was kept, where roads typically occur. This prior knowledge further removes the unnecessary interference items and noises, simplifying the complexity and enabling the model to focus on the road in the next step.
5. Implemented Hough Transform to find the road, since the boundaries are normally a series of lines. Since the image has been cropped, the edge of the cropping may also be identified as lines. Therefore, a buffer length was set to filter the lines close to the boundaries, avoiding the misdetection. The end points of line segments were stored for further processing.
6. A convex hull of the collected end points was computed. This helps to form a closed polygon to cover the whole road area, i.e. the final road detection result.
7. Blending was used to overlay the road detection result onto the original image for better visualization and evaluation.
8. For a more reasonable quantitative evaluation, the IoU between the ground truth and the prediction was computed: $IoU = |GT \cap Prediction|/|GT \cup Prediction|$ .

#### Details

1. In the preprocessing step, for Gaussian blurring, the kernel size is $5\times5$ and $\mu=0$. For Canny edge detection, the lower threshold is $T_{low}=50$ and the higher threshold is $T_{high}=150$. When selecting the region of interest (ROI), the bottom-left vertex ( $(height, 0)$ ), the bottom-right vertex ( $(height, width)$ ), and the center point of the image ( $(height/2, width/2)$ ) are selected, and the triangle area enclosed is the ROI, where the road normally appear. A mask is applied using bitwise-and to crop the ROI and generate the final preprocessed image.
2. In Hough Transform, the following parameters are set to detect the lines in the preprocessed edge image: $\rho=1, \theta=\pi/180, threshold=50, min \textunderscore line \textunderscore length=100, max \textunderscore line \textunderscore gap=50$ . To avoid misinterpreting the boundaries of ROI, which may have prominent edge with respect to the masked area, a buffer width $buffer=10$ is set to filter out the lines within the buffer width to the boundaries.
3. After finding all lines related to the road, a convex hull is found to fill in the road area, and the weights of blending are $w_{image}=0.7$ for the original image and $w_{road}=0.3$ for the road segmentation result.

#### Results
#####  Randomly Chosen Image "umm_000061.png"
![image](https://github.com/user-attachments/assets/a7a03cdc-5add-4233-a282-573bd445f50c)

##### Image After Preprocessing (Grayscaling and Gaussian Blurring)
![image](https://github.com/user-attachments/assets/3f5776c3-bd61-46cb-9a8c-4a11a4c57281)

##### ROI of the Preprocessed Image
![image](https://github.com/user-attachments/assets/1bae51d5-e5e9-4e1e-a369-f2da98aeb6cb)

##### Line Detection Result by Hough Transform
![image](https://github.com/user-attachments/assets/f4d08029-48d8-46d5-857a-9ebd3e62716f)

##### Road Detection Result by Convex Hull
![image](https://github.com/user-attachments/assets/c28d581b-157d-4466-b877-bbe51aaffd99)

##### Road Segmentation Ground Truth
![image](https://github.com/user-attachments/assets/295c3dca-d2ef-40bf-a42c-2b81d3600f13)

##### Road Detection Result Blended with Original Image
![image](https://github.com/user-attachments/assets/09ac454e-14d8-4ec5-b2b2-b51992819249)

##### Road Detection Ground Truth Blended with Original Image
![image](https://github.com/user-attachments/assets/a452dfe0-21af-4701-b006-6784768c69fb)

##### Intersection Map between Ground Truth and Prediction
![image](https://github.com/user-attachments/assets/0d942fff-8a76-49dd-9923-d12ef4d87737)

##### Union Map between Ground Truth and Prediction
![image](https://github.com/user-attachments/assets/ae664c2f-db24-40af-8de6-e86ce1d22ef0)

##### Quantitative Evaluation
$IoU = 0.6978$

#### How to use

The code for the first update is available in "project.ipynb". You can see the results I get in my implementation by directly going through the file. Besides, to replicate the result, you can do the following:

1. Install the required packages:
   - opencv-python
   - numpy
   - matplotlib
2. Download the KITTI Road Dataset from [https://www.cvlibs.net/datasets/kitti/eval_road.php](https://www.cvlibs.net/datasets/kitti/eval_road.php), create a new folder named "dataset" and place the downloaded folder in "dataset" folder. You are also free to customize the dataset by editing the image path from code block 2 to 4.
3. Go to "project.ipynb" and run the code.

#### Next Step
1. Further improve the road detection result by integrating some new features.
2. Detect and track objects like vehicles or pedestrians in each frame. Methods to be used and integrated include SIFT, optical flow and neural networks like YOLO.
3. Test the model on a complete video containing all frames and build an interface displaying the result.
