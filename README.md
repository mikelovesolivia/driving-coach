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

### Second Update

#### U-Net for Lane Segmentation

The model is trained on the training split of the KITTI road dataset. Only images with a lane segmentation map (95 in total) are chosen. The training split is randomly divided into 2 datasets:
80% (76) for the training dataset and 20% (19) for the validation dataset. 

To detect lanes from the images, I implemented a U-Net and trained it with image and mask pairs from the training dataset.

The structure of the U-Net model is shown in the following image:

![image](https://github.com/user-attachments/assets/f1e613ad-cdc6-4888-ba99-c4a30bfcd848)

First, each input image is resized to $256 \times 512$ to be in the same dimension. Then each pixel value of the image is divided by 255 and normalized to $[0, 1]$. Afterwards, the image is noramlized by $\mu=[0.485, 0.456, 0.406], \sigma=[0.229, 0.224, 0.225]$ for each channel and input into the U-Net model. The U-Net model consists of several convolutional blocks, transposed convolution layers, max pooling layers, skip connections and concatenations. The convolutional block consistss of a convolutional layer that preserves the input size, a batch norm layer and a ReLU activation function. The max pooling layer downsamples the feature maps by 2. Finally, the model outputs a segmentation map of the lanes. For optimization, I binarized the ground truth segmentation map, computed the binary cross entropy loss 

$\text{BCE Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left( y_i \cdot \log(\hat{y}_i) + (1 - y_i) \cdot \log(1 - \hat{y}_i) \right)$

and optimized the model with Adam optimizer. Learning rate is set to $10^{-4}$.

The U-Net is used in this segmentation task because:
- High-resolution features from the contracting path are combined with the upsampled output in the expansive path, enhancing localization. A successive convolutional layer refines this information to assemble a more precise segmentation output.
- The large number of feature channels in the upsampling part enables the network to effectively propagate context information to higher-resolution layers, ensuring detailed and contextually accurate segmentation.
- The symmetrical architecture facilitates the seamless integration of low-level spatial information from the contracting path with high-level semantic information from deeper layers, resulting in a balanced and comprehensive feature representation.
- Skip connections between corresponding layers in the contracting and expansive paths preserve fine-grained details and help mitigate the loss of spatial resolution during downsampling.
- The multi-scale processing capability of U-Net allows it to capture both global context and local details, making it robust for segmenting objects of varying sizes.


After 20 epochs of training, three evaluation metrics (on training set only since the test set does not contain ground truth segmentation masks) are selected: 

- **Dice Coefficient**:
  - The Dice Coefficient measures the overlap between the predicted lane pixels and the ground truth lane pixels.
  - Dice prioritizes the overlap between the predicted and actual lane pixels, which is crucial for ensuring that the detected lanes align accurately with the true lane boundaries.
  - $\text{Dice Coefficient} = \frac{2 \cdot |P \cap G|}{|P| + |G|}$

- **IoU**:
  - The IoU measures the ratio of the intersection area to the union area between the predicted and ground truth masks.
  - IoU ensures that the predicted lanes do not include irrelevant regions, as it penalizes both false positives (extra regions that are misclassified) and false negatives (missed lane regions).
  - $\text{IoU} = \frac{|P \cap G|}{|P \cup G|}$
 
- **Pixel Accuracy**:
  - The Pixel Accuracy calculates the proportion of correctly classified pixels (both lane and non-lane) to the total number of pixels.
  - Pixel accuracy provides a general measure of how well the model distinguishes between lane and non-lane pixels.
  - $\text{Pixel Accuracy} = \frac{\text{Number of Correctly Classified Pixels}}{\text{Total Number of Pixels}}$



Evaluation metrics are displayed in the table below:

| Metric                     |  Train  |   Val   |
|----------------------------|---------|---------|
| Average Dice Coefficient   | 0.9591  | 0.9526  |
| Average IoU                | 0.9215  | 0.9102  |
| Average Pixel Accuracy     | 0.9935  | 0.9923  |



And the ROC and PR Curves are plotted in the figures below:

Training Set:

![image](https://github.com/mikelovesolivia/driving-coach/blob/main/resources/roc_train.png)

![image](https://github.com/mikelovesolivia/driving-coach/blob/main/resources/pr_train.png)


Validation Set:

![image](https://github.com/user-attachments/assets/b673b34a-a6d7-4f72-abf1-c666cb77328c)


![image](https://github.com/user-attachments/assets/c78dbf6f-2723-4d4b-9715-2481a8d8a975)


Which indicates that the model fits the training data and generalized to the validation data well.

In prediction, only pixels with probability greater than 0.5 are selected as the lane segmentation. Then the segmentation results are overlaid with the original images to provide a more intuitive visualization.

Below are some selected segmentation results:

##### Training set
![image](https://github.com/user-attachments/assets/3347f430-6fed-4842-ae99-94562e5a9e0d)

![image](https://github.com/user-attachments/assets/d153a464-ba33-4a0a-b662-1c7f71ac3430)

![image](https://github.com/user-attachments/assets/7c727eeb-b9a4-4754-897c-20c52cf64bf3)

![image](https://github.com/user-attachments/assets/7ac2dc90-a6c0-464f-9be5-d189fab4dfdd)

##### Validation set
![image](https://github.com/user-attachments/assets/daf3c471-33bc-4fc6-bd6f-38cb41a1bb32)

![image](https://github.com/user-attachments/assets/999d8a79-a921-47f6-86ae-3e96bd57a9c6)

![image](https://github.com/user-attachments/assets/c830750c-885b-4f67-82e5-6cee7ae0bfb1)

![image](https://github.com/user-attachments/assets/5610b9a8-2c0c-45f4-880d-5301f793419f)


Most test results make sense. Some cases are not perfectly well due to light conditions and shades.

I also apply the model to a [video](https://github.com/mikelovesolivia/driving-coach/blob/main/resources/straight_lane_detected_notprocessed.avi), which turns out not to work very well. This may attribute to that the aspect ratios (height/width) of the video and training images vary greatly, and I simply apply the resize for the input to be processed by the model. A cropping operation may be better.

To compare, I also test the deeplabv3_resnet50 model imported from torchvision.models, and it turns out to work better on the test video. It may arise from its unique structure design:
- Multi-scale Context Aggregation: Atrous convolutions and ASPP enable efficient capture of global and local features for lanes of varying shapes and sizes.
- High Spatial Resolution: Retains fine details with Atrous convolutions and avoids excessive downsampling.
- Global Context Encoding: Incorporates image-level features for improved robustness in challenging scenarios.

Therefore, the followings may be applied to the U-Net architecture for better performance:
- Introduce multi-scale context by adding ASPP or atrous convolutions for better multi-scale feature extraction.
- Preserve spatial resolution by replacing aggressive downsampling with atrous convolutions.
- Incorporate global context and use image-level features to improve robustness in complex scenarios.
- Apply brightness, contrast, and gamma corrections during training to make the model robust to different light conditions.


### Final Update

### Dataset

The test dataset is the test split of the KITTI road dataset. There are 179 images in total. Similarities between images in training/validation and test set like the similar aspect ratio anf resolution of the image, the time when the images are taken (all during daytime; no images taken at night), and the color of the road (gray with varying light and shades) ensure that the features the model learns from the training set can be applied to the validation set. Moreover, due to the differences of each image in the dataset, including the position and the shape of the lane, the light condition, the direction of the road, the variations of the background scene, the shades on the road, etc., this division of training and validation set is effective. Such differences allow for different factors that should be considered during lane detection, sufficiently enabling the model to be evaluated for its generalization ability on the test set. Furthermore, I also picked two videos (one straight lane and the other curved lane) to test the model. Compared with the training set where the background scene mostly consists of trees and houses, the test video demonstrates a drive on the highway. Besides, different from images, video exhibits continuity and similarities across frames. These differences help assess the model's performance in a real driving setting: whether it still predicts the right lanes under different background scenes, achieves good results when lane shapes are different (straight or curved) and preserves the continuity in continuous frames.

There are no ground truth labels for the test split of KITTI road dataset. Therefore, for evaluation, only qualitative results can be reported. Here are some results from the test set:

![image](https://github.com/mikelovesolivia/driving-coach/blob/main/resources/test/unet/um_000017.png)
Test Fig. 1

![image](https://github.com/mikelovesolivia/driving-coach/blob/main/resources/test/unet/um_000024.png)
Test Fig. 2

![image](https://github.com/mikelovesolivia/driving-coach/blob/main/resources/test/unet/uu_000036.png)
Test Fig. 3

![image](https://github.com/mikelovesolivia/driving-coach/blob/main/resources/test/unet/uu_000037.png)
Test Fig. 4

![image](https://github.com/mikelovesolivia/driving-coach/blob/main/resources/test/unet/uu_000049.png)
Test Fig. 5

![image](https://github.com/mikelovesolivia/driving-coach/blob/main/resources/test/unet/uu_000050.png)
Test Fig. 6

![image](https://github.com/mikelovesolivia/driving-coach/blob/main/resources/test/unet/uu_000051.png)
Test Fig. 7

![image](https://github.com/mikelovesolivia/driving-coach/blob/main/resources/test/unet/uu_000053.png)
Test Fig. 8

Here is a GIF to show all results on the test dataset:

![image](https://github.com/mikelovesolivia/driving-coach/blob/main/resources/test/unet/test.gif)

Most results seem reasonable on the test set. However, there are some less satisfying results. For example, in Test Fig. 3, 4 and 7, the segmentation covers too little area. This may be due to that there are no white lane lines guiding the model to locate at the right place. Besides, in Test Fig. 8, the car's color is closer to the gray roads in the training samples, misleading the model to recognize it as the lane. 

Also, I tested the model on two videos, one [straight lane](https://github.com/mikelovesolivia/driving-coach/blob/main/straight_lane.mp4) and the other [curved lane](https://github.com/mikelovesolivia/driving-coach/blob/main/curved_lane.mp4). Based on the insight from the last update, I made a comparison of [1] directly inputting the resized frame to the model and [2] cropping the frame to the similar aspect ratio as the training images, then resizing the frame and inputting to the model. While Method [1] shows bad results with distorted lane shape with several broken areas, Method [2] demonstrates much better results. This shows that the aspect ratio is also an important factor that affects the model performance: inconsistent aspect ratio introduces distortions and noises in the segmentation results. Besides, the model predicts better lane segmentations on straight lanes than curved lanes, which may be attributable to that (1) the training set contain more straight lane samples; (2) straight lane is more regular than curved lane, therefore easier to segment. Overall, the model also produces good qualitative results on the test video, showcasing its generalization ability on continuous frames with background scenes different to the training setting. Below are the links to the videos:

|               |  Method [1]  |  Method [2]  | Ground Truth |
|---------------|--------------|--------------|--------------|
| Curved Lane   | [video](https://github.com/mikelovesolivia/driving-coach/blob/main/resources/curved_lane_detected_notprocessed.avi)  | [video](https://github.com/mikelovesolivia/driving-coach/blob/main/resources/curved_lane_detected_processed.avi)  | [video](https://github.com/mikelovesolivia/driving-coach/blob/main/curved_lane.mp4)  |
| Straight Lane | [video](https://github.com/mikelovesolivia/driving-coach/blob/main/resources/straight_lane_detected_processed.avi)    | [video](https://github.com/mikelovesolivia/driving-coach/blob/main/resources/straight_lane_detected_notprocessed.avi)  |  [video](https://github.com/mikelovesolivia/driving-coach/blob/main/straight_lane.mp4) |

To improve the results, here are some proposed solutions:

- **Use more training samples.** KITTI dataset contains only small number of labeld images, which are too little to train a powerful model to generalize to more complicated settings. Use a larger dataset for training can help improve the performance.
-  **Employ data augmentation.** Test results show that factors like light and color can misguide the model. Image augmentation methods, like color jitter, gray scaling, gamma correction, histogram equalization can be used to prevent the model from overly depending on such misguiding factors. Besides, with the aforementioned image augmentation methods and geometric transformations (flip, rotation, translation, crop, perspective warpping, affine transformations, etc.), more image-label pairs for training can be generated, expanding the training dataset. Adding some noises like Gaussian noise, salt-and-pepper noise, Poisson noise, etc. can also improve the model's robustness. 
-  **Integration with other methods.** We can also guide the model with some prior knowledge extracted by some other methods, like hough transform, RANSAC, etc., to constrain the model's focus on places where there might be lanes, therefore improving the performance.
-  **More powerful models** More powerful models, like Vision Transformers, YOLO, etc. may generate better results.

### Presentation Slide

You can find my slide in this [link](https://docs.google.com/presentation/d/17ztJfamhMHe-j9Yk8N3kGRjkXg4tZRlNP_dlqvxtAG0/edit#slide=id.p1).

### Running instructions:

1. Download the [KITTI road dataset](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_road.zip), create a folder named "dataset", and place the "data_road" folder in it.
2. Install the required packages by running
```
pip install -r requirements.txt
```  
3. Run "unet.ipynb" line by line. ("project.ipynb" is a comparison experiment that uses hough transform; you can also run it line by line for comparison.)
4. Following results will be generated:
   - train_results: this folder contains all results on the training dataset
   - val_results: this folder contains all results on the validation dataset
   - test_results: this folder contains all results on the test dataset
   - test.gif: a gif image showing all results on the test dataset for an intuitive and convenient visualization
   - curved_lane_detected_processed.avi: the detection result on a video which shows a car driving on a road with curved lanes, created by Method [2] mentioned in the final update.
6. If you want to use the model I trained for evaluation, uncomment "model.load_state_dict(torch.load(f"unet_epoch_20.pth"))" in the 1st line of the 7th block in "unet.ipynb".








