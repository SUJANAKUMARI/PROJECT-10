# PROJECT-10

## CUSTOM OBJECT CHARACTER RECOGNITION

# OBJECTIVE:

To build a Custom OCR by combining YOLO and Tesseract, to read the specific contents of a Lab Report and convert it into an editable file. Use YOLO_V3 to train on the personal dataset. Then the coordinates of the detected objects are passed for cropping the detected objects and storing them in another list. This list is passed through the Tesseract to get the desired output. 

# INSTALL REQUIRED LIBRARIES:  
import streamlit as st
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import pytesseract
import cv2
import os

# DEPENDENCIES: 

● Install Tesseract OCR Engine in the system 
● Install Pytesseract library pip install pytessercat 
● Install OpenCV pip opencv

# RESIZING THE IMAGES AND LABELLING

The first step of the process is taking the bounding box coordinates from YOLOv3 and simply taking the region within the bounds of the box. As this image is super small, we use cv2.resize() to blow the image up 3x its original size.
 
Then I labelled the images by creating bounding boxes consisting of 4 classes i.e., TestName, Value, Units and NormalRange from which I got the labelled images in yolo format with respective file names in labels folder.

# TRAINING THE YOLO MODEL

After creating the labels and getting the labels for each image, I trained the using Yolo3 architecture named Yolov3.pt

Below is the code for training the model trained using 100 epochs, 16 batches, 640 image size, and named yolov3_train.

After training the yolo model I got weights named as best.pt.

![image](https://github.com/user-attachments/assets/66065e9c-c612-413b-a535-dc13c9a13187)

# OUTPUT OF THE TRAINED MODEL

# CONFUSION MATRIX:
![image](https://github.com/user-attachments/assets/163f02a6-b585-4429-a10e-4daeb0979797)

# F1 CONFIDENCE CURVE
![image](https://github.com/user-attachments/assets/aebab2f2-aaa5-4c23-980f-97cc4484101f)

# LABELS
![image](https://github.com/user-attachments/assets/3abe2994-ba3d-4da1-a712-620897268996)

# LABEL CORELOGRAM
![image](https://github.com/user-attachments/assets/257999d3-f55f-4f02-9734-c03e8a2adfd1)

# P-CURVE
![image](https://github.com/user-attachments/assets/76a1e96f-8362-47c7-aedc-96868318910c)

# R-CURVE
![image](https://github.com/user-attachments/assets/976d7895-262b-4916-a821-23e931ac0a04)

# PR CURVE
![image](https://github.com/user-attachments/assets/2f982b47-70f8-4999-b941-39dafcafd8c9)

# BOUNDING BOX LOSS, CLASSIFICATION LOSS AND OTHER METRICS
![image](https://github.com/user-attachments/assets/70d9b225-91d9-4f88-87c8-fc32647dfe62)

# OCR RECOGNITION OF THE TRAINED MODEL
![image](https://github.com/user-attachments/assets/9b899d39-6195-4f13-a697-12dc20c6d046)
![image](https://github.com/user-attachments/assets/c06d46b0-d4ed-449b-9bdb-a8d0a2def8c0)

# TEXT DETECTION USING TESSERACT  AND DEPLOYMENT USING STREAMLIT
Named the file as deploy.py and below are the steps explaining it one by one

Load the YOLO Model:  The trained YOLO model (best.pt) is loaded to detect text regions in the image.  

User Uploads an Image:   The Streamlit UI allows users to upload images (jpg, jpeg, png).

Text Detection & Extraction:  The YOLO model identifies text regions.  Each detected region is cropped, converted to grayscale, and processed with Tesseract OCR to extract readable text.

Displaying Results:  The original image is displayed first.  

Then, the processed image is shown with bounding boxes around detected text. 

 Finally, the extracted text is displayed in a readable format.

Here is the screenshot of the streamlit interface page.
![image](https://github.com/user-attachments/assets/8f90f6c7-a996-4fa9-a796-1af0517503b5)

Here is the video of testing of the streamlit interface and how it gives the output.
[Watch the video] https://github.com/SUJANAKUMARI/PROJECT-10/blob/main/Deployment%20screen%20recording.webm

SCREENSHOT OF THE OUTPUTS
![image](https://github.com/user-attachments/assets/854dffa0-987a-4892-a08a-38bd4ce26d4b)
![image](https://github.com/user-attachments/assets/8e0ff0b9-1cf2-495f-befc-30e94cb1d503)
![image](https://github.com/user-attachments/assets/d89c9260-ed4c-49e5-bca0-bdd2444a5f9e)
![image](https://github.com/user-attachments/assets/fda5ded7-a41e-41d3-b8bc-0f042d18d9b6)


### PERFORMANCE METRICS EXPLANATION:

## PATH:  📂 YOLOv3_Custom/yolov3_train/results.csv

During the training and evaluation of our YOLOv3 model, several key performance metrics were recorded to assess the model’s effectiveness in detecting and recognizing objects:

## 1. Training & Validation Losses:
train/box_loss & val/box_loss – Measures how accurately bounding boxes are predicted. Lower values indicate better performance.

train/cls_loss & val/cls_loss – Represents classification loss, determining how well the model distinguishes different objects.

train/dfl_loss & val/dfl_loss – Represents the Distribution Focal Loss, used to refine box regression.

## 2. Detection Performance Metrics:
metrics/precision(B) – Precision measures how many of the detected objects are actually correct (i.e., how precise the model is in detecting only relevant objects).

metrics/recall(B) – Recall measures how many of the actual objects present in the image were correctly detected by the model.

metrics/mAP50(B) – Mean Average Precision at 50% Intersection over Union (IoU) threshold. This is a standard metric in object detection, evaluating how well the model predicts bounding boxes.

metrics/mAP50-95(B) – Mean Average Precision across different IoU thresholds (from 50% to 95%). A more comprehensive evaluation metric.

## 3. Learning Rate Tracking:
lr/pg0, lr/pg1, lr/pg2 – Represents the learning rates of different layers in the model, helping monitor how the optimizer updates the model during training.


## ACKNOWLEDGMENT:

## I would like to express my sincere gratitude to Ms. Twinkle Baid for her invaluable guidance and support throughout this project.
