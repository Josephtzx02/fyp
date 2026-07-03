# 3D Scene Analysis and Robotic Vision for Intelligent Book Retrieval Systems

> Final Year Project (FYP)
> Bachelor of Manufacturing Engineering with Management (Hons.)
> Universiti Sains Malaysia (USM)

## Overview

This repository contains the complete source code developed for my Final Year Project titled:

**"3D Scene Analysis and Robotic Vision for Intelligent Book Retrieval Systems"**

The project proposes an RGB-D robotic vision pipeline capable of:

* Detecting book spines using **YOLOv8n-Oriented Bounding Box (YOLOv8n-OBB)**
* Estimating real-world 3D coordinates using an **Intel RealSense D435**
* Predicting hidden physical properties (book width and weight) using Machine Learning regression models
* Producing a unified perception output for future robotic grasping and automated library retrieval systems

The complete methodology is described in my thesis, while this repository contains all source code used throughout the development process.

---

# Overall Project Workflow

```
Data Collection
      │
      ▼
Dataset Annotation (Roboflow)
      │
      ▼
Annotation Filtering & OBB Conversion
      │
      ▼
YOLOv8n-OBB Training
      │
      ▼
Book Width & Weight Dataset Collection
      │
      ▼
Regression Model Training
      │
      ▼
Real-Time Deployment
```

---

# Repository Structure

```
.
├── pkl_files_generated/
├── others_nonused_codes/
├── data_capture/
├── subfeatures/
├── datafilter/
│
├── train_model_width_weight.py
├── realtime_fps_test.py
├── obb_clickable_demo.py
├── weight_dataset.csv
├── data.yaml
├── yolov8n-obb.pt
├── best.pt
└── README.md
```

---

# Main Files

## train_model_width_weight.py

This script trains the machine learning regression models used for estimating the hidden physical properties of books.

It trains:

* Book width prediction model
* Book weight prediction model

After training, four serialized `.pkl` files are automatically generated and stored inside the **pkl_files_generated** folder.

---

## realtime_fps_test.py

Utility script used to benchmark the inference speed of the trained YOLOv8n-OBB detector.

Purpose:

* Load `best.pt`
* Run real-time detection using Intel RealSense D435
* Measure inference FPS
* Evaluate deployment performance

---

## obb_clickable_demo.py

**Main deployment program of the entire project.**

This is the final real-time application demonstrated in the project.

Functions include:

* Real-time RGB-D acquisition from Intel RealSense D435
* Book spine detection using YOLOv8n-OBB (`best.pt`)
* 3D coordinate extraction
* Book selection through clickable interface
* Width prediction
* Weight prediction
* Display of detected book parameters in real time

This program loads:

```
best.pt
```

and

```
width_model.pkl
width_features.pkl
weight_model_D_huber.pkl
weight_features_D.pkl
```

to perform complete inference.

---

## weight_dataset.csv

Contains the complete regression dataset consisting of **205 books**.

Each record includes ground-truth measurements such as:

* Height
* Thickness
* Width
* Weight

along with engineered features used for machine learning model training.

---

## data.yaml

YOLOv8 dataset configuration file.

Defines:

* training dataset
* validation dataset
* testing dataset
* class names

Used during YOLOv8-OBB training.

---

## yolov8n-obb.pt

Official pretrained YOLOv8 Nano Oriented Bounding Box model provided by Ultralytics.

Used as the initial pretrained backbone before custom training.

---

## best.pt

Final trained YOLOv8n-OBB model obtained after training.

The usual path looks like this: runs/obb/train3/weights/best.pt

This repository uses the checkpoint with the best validation performance (Epoch 111) for deployment.

---

# Folder Description

---

# 1. pkl_files_generated/

Contains the trained regression models generated after executing:

```
train_model_width_weight.py
```

Files:

```
weight_features_D.pkl
weight_model_D_huber.pkl
width_features.pkl
width_model.pkl
```

These files are loaded during real-time deployment:

```python
width_model = joblib.load("width_model.pkl")
width_features = joblib.load("width_features.pkl")

weight_model = joblib.load("weight_model_D_huber.pkl")
weight_features = joblib.load("weight_features_D.pkl")
```

### Description

**width_model.pkl**

Trained regression model used to predict the hidden width of a book.

---

**width_features.pkl**

Stores the selected feature list used during width model training, ensuring identical feature ordering during deployment.

---

**weight_model_D_huber.pkl**

Final Huber Regression model (Model D) used to estimate book weight from visual and geometric features.

---

**weight_features_D.pkl**

Stores the selected feature configuration for the deployed weight prediction model.

---

# 2. others_nonused_codes/

Contains previous experimental scripts developed throughout the research.

Examples include:

* regression_step1_vision
* regression_step2AB_kmeans
* regression_step5AB_pixelanddepth
* train_width
* train_weight
* train_width_weight
* boundingbox
* predictwidth
* predictwidthhistogram
* predictwidthClandPWT
* calibrated Dvalue
* checkfixed

These scripts were created during experimentation, algorithm evaluation, and prototype development.

They are **not required** for the final deployment pipeline but are retained as references for the research and development history.

---

# 3. data_capture/

Contains all scripts used for dataset acquisition.

---

## fyp_capture_tool.py

The primary data acquisition program.

This is the **first step** of the entire project.

The script captures synchronized RGB-D data from the Intel RealSense D435 and automatically saves the dataset into:

```
RGB/
DEPTH_RAW/
DEPTH_COLORMAP/
META/
INTRINSICS/
```

The captured RGB images are later uploaded to Roboflow for annotation before YOLOv8-OBB training.

---

## capture_cnn_data.py

Used after the object detector has been trained.

The script loads the trained `best.pt` model and allows the user to select individual detected books from the camera view.

Ground-truth information including width and weight can then be entered manually to build the regression dataset used for machine learning.

This script was used to create the final **205-book regression dataset**.

---

## capture_cnn_data_v1.py

Earlier version of the data collection tool before weight collection functionality was introduced.

Maintained only for reference.

---

## capture_cnn_data_depthonly.py

Specialized data collection tool for experiments involving fixed calibrated depth measurements.

---

## capture_cnn_data_focus.py

Dataset collection tool designed for targeted sampling of specific book categories.

Rather than collecting books randomly, the script prioritizes underrepresented structural groups such as:

* small pocket books
* large and tall books
* thick and heavy books
* very thin books with normal height

This helps improve dataset diversity and balance during regression model development.

---

# 4. subfeatures/

Contains supporting utilities used throughout the project.

These scripts assist with visualization, validation, debugging, and data analysis.

Examples include:

* plot4.3
* plot4.4.3
* normal_distribution
* depth_measure
* visual_check
* testd435
* autoclicker

These scripts were mainly used to:

* generate thesis figures
* visualize regression performance
* inspect captured data
* verify camera measurements
* perform hardware testing
* automate repetitive experimental procedures

---

# 5. datafilter/

Contains the complete preprocessing pipeline used before YOLOv8-OBB training.

This stage converts Roboflow annotations into clean YOLOv8-OBB labels.

---

## convert_to_obb.py

Converts Roboflow polygon annotations into standardized Oriented Bounding Boxes.

Processing includes:

* Ramer-Douglas-Peucker (RDP) polygon simplification
* Convex Hull generation
* OpenCV minimum-area rectangle fitting
* Clockwise corner ordering

The output is a refined OBB annotation file suitable for YOLOv8-OBB training.

---

## convert_json_to_labels.py

Converts the processed COCO JSON annotation file into YOLOv8-OBB label files.

The script:

* reads every annotated polygon
* normalizes coordinates using image width and height
* converts pixel coordinates into YOLO format
* generates one `.txt` label file for every image

Each label follows the format:

```
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

which is required by the Ultralytics YOLOv8-OBB training framework.

---

## visual_check_obb.py

Visualization tool used to verify annotation quality.

It overlays:

* **Red:** Original Roboflow polygons
* **Green:** Refined OBB annotations

This provides a quick visual comparison to ensure preprocessing has correctly simplified and aligned the bounding boxes before training.

---

## rename.py

Simple utility used to rename annotation files downloaded from Roboflow so that filenames match the project's dataset organization.

---

# Dependencies

Major libraries used:

* Python 3.x
* Ultralytics YOLOv8
* OpenCV
* Intel RealSense SDK (pyrealsense2)
* NumPy
* Pandas
* Scikit-learn
* Joblib
* Matplotlib

---

# Hardware

* Intel RealSense D435 RGB-D Camera
* NVIDIA RTX 4050 Laptop GPU
* NVIDIA RTX A2000 Workstation GPU

---

# Citation

If you use this repository in your research, please cite:

**Joseph Teh Zhe Xi**

*3D Scene Analysis and Robotic Vision for Intelligent Book Retrieval Systems*

Bachelor of Manufacturing Engineering with Management (Hons.)

Universiti Sains Malaysia, 2026.

---

# License

This repository is shared for academic and educational purposes.

Please provide appropriate attribution if any part of the source code or methodology is used in future research.
