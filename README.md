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
├── training_setup/
├── data_capture/
├── datafilter/
├── pkl_files_generated/
├── subfeatures/
├── others_nonused_codes/
│
├── obb_clickable_demo.py
├── train_model_width_weight.py
├── realtime_fps_test.py
├── best.pt
└── README.md
```

---

# Quick Start

The complete workflow to reproduce this project is summarized below.

### Stage 1 — Capture RGB-D Dataset

Run:

```text
data_capture/fyp_capture_tool.py
```

This captures synchronized RGB-D images from the Intel RealSense D435. The captured RGB images are later uploaded to Roboflow for annotation.

---

### Stage 2 — Annotate Images

Upload the captured RGB images to Roboflow and annotate the book spines using polygon segmentation.

Export the annotated dataset in **COCO Segmentation** format.

---

### Stage 3 — Generate and Verify OBB Annotations

Inside `datafilter/`:

1. Run `rename.py` if the exported filenames need to be renamed to match the project dataset structure.
2. Run `convert_to_obb.py` to convert the Roboflow polygon annotations into refined Oriented Bounding Boxes (OBBs). This script performs:
   - Ramer-Douglas-Peucker (RDP) polygon simplification
   - Convex Hull generation
   - OpenCV minimum-area rectangle fitting
   - Clockwise corner ordering
3. (Optional but recommended) Upload the generated OBB annotation file to a new Roboflow project for visual verification. This allows incorrect or poorly fitted bounding boxes to be inspected and corrected before model training.
4. Export the verified dataset in **COCO Segmentation** format.
5. Run `convert_json_to_labels.py` to convert the exported COCO annotation file into YOLOv8-OBB label files.
---

### Stage 4 — Train YOLOv8-OBB

Follow the instructions in `training_setup/README.md`.

The primary training command is:

```bash
yolo task=obb mode=train model=yolov8n-obb.pt data=data.yaml epochs=200 imgsz=640 batch=8
```

After training completes, copy the generated `best.pt` checkpoint from:

```text
runs/obb/train/weights/best.pt
```

into the project root directory.

---

### Stage 5 — Collect Regression Dataset

Run:

```text
data_capture/capture_cnn_data.py
```

Using the trained detector, manually record the ground-truth width and weight of the selected books to construct the regression dataset.

---

### Stage 6 — Train Regression Models

Run:

```text
train_model_width_weight.py
```

This generates the trained regression models stored in `pkl_files_generated/`.

---

### Stage 7 — Run the Complete System

Run:

```text
obb_clickable_demo.py
```

The program loads:

- `best.pt`
- `width_model.pkl`
- `width_features.pkl`
- `weight_model_D_huber.pkl`
- `weight_features_D.pkl`

to perform real-time book detection, 3D localization, width prediction, and weight prediction.

---

---

# Main Files

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

## train_model_width_weight.py

This script trains the machine learning regression models used for estimating the hidden physical properties of books.

It trains:

* Book width prediction model
* Book weight prediction model

After training, four serialised `.pkl` files are automatically generated and stored inside the **pkl_files_generated** folder.

---

## realtime_fps_test.py

Utility script used to benchmark the inference speed of the trained YOLOv8n-OBB detector.

Purpose:

* Load `best.pt`
* Run real-time detection using Intel RealSense D435
* Measure inference FPS
* Evaluate deployment performance

---

## best.pt

Final trained YOLOv8n-OBB model obtained after training.

The usual path looks like this: runs/obb/train/weights/best.pt

This repository uses the checkpoint with the best validation performance (Epoch 111) for deployment.

---

# Folder Description

## 1. training_setup/

Contains the files and documentation required to reproduce the custom YOLOv8n-Oriented Bounding Box (YOLOv8n-OBB) training pipeline used in this project.

This folder includes:

### README.md

A step-by-step guide for setting up the Python environment, installing the required dependencies, preparing the dataset, training the YOLOv8n-OBB model from scratch, resuming interrupted training, running inference, and benchmarking model performance.

### data.yaml

YOLOv8 dataset configuration file.

Defines:

- Training dataset
- Validation dataset
- Testing dataset
- Class names

Used during YOLOv8-OBB training.

### yolov8n-obb.pt

Official pretrained YOLOv8 Nano Oriented Bounding Box model provided by Ultralytics.

Used as the transfer learning backbone before custom training on the book spine dataset.

---

## 2. data_capture/

Contains all scripts used for dataset acquisition throughout the project.

### fyp_capture_tool.py

The primary RGB-D data acquisition program and the first step of the complete project workflow.

This script captures synchronized RGB and depth images from the Intel RealSense D435 and automatically stores them into the following folders:

```
RGB/
DEPTH_RAW/
DEPTH_COLORMAP/
META/
INTRINSICS/
```

The captured RGB images are later uploaded to Roboflow for annotation before YOLOv8-OBB training.

### capture_cnn_data.py

Used after the YOLOv8-OBB detector has been trained.

The script loads the trained `best.pt` model and allows users to select detected books directly from the camera view.

Ground-truth measurements, including book width and weight, can then be manually entered to build the regression dataset used for machine learning.

This script was used to generate the final dataset consisting of **205 books**.

### capture_cnn_data_v1.py

Earlier version of the regression dataset collection tool before weight collection functionality was incorporated.

Retained for reference only.

### capture_cnn_data_depthonly.py

Specialized data collection tool used for experiments involving calibrated fixed-depth measurements.

### capture_cnn_data_focus.py

Dataset collection tool designed to improve dataset diversity by prioritizing underrepresented categories of books instead of random sampling.

Examples include:

- Small pocket books
- Large and tall books
- Thick and heavy books
- Very thin books with normal height

This strategy helps produce a more balanced regression dataset.

## width_weight_dataset.csv

Contains the complete regression dataset consisting of **205 books**.

Each record includes ground-truth measurements such as:

* Height
* Thickness
* Width
* Weight

along with engineered features used for machine learning model training.

---

## 3. datafilter/

Contains the complete preprocessing pipeline used before YOLOv8-OBB model training.

These scripts convert Roboflow polygon annotations into clean YOLOv8-OBB labels.

### convert_to_obb.py

Converts Roboflow polygon annotations into refined Oriented Bounding Boxes (OBBs).

The script performs the following preprocessing pipeline:

- Ramer-Douglas-Peucker (RDP) polygon simplification
- Convex Hull generation
- OpenCV minimum-area rectangle fitting
- Clockwise corner ordering

The processed annotations are exported as an OBB-formatted COCO annotation file. The generated file may optionally be uploaded to Roboflow for manual verification before generating the final YOLOv8-OBB labels.

### convert_json_to_labels.py

Converts the verified COCO annotation file into YOLOv8-OBB label files required by the Ultralytics training framework.

The script:

- Reads each verified OBB annotation
- Normalizes the corner coordinates using the corresponding image dimensions
- Converts the coordinates into YOLOv8-OBB format
- Generates one `.txt` label file for each image

Each label follows the format:

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

which is required by the Ultralytics YOLOv8-OBB training framework.

### visual_check_obb.py

Visualization tool used to compare original and simplified annotations.

- **Red:** Original Roboflow polygons
- **Green:** RDP-simplified polygons

This provides a quick visual verification that polygon simplification has been correctly applied before generating the final Oriented Bounding Boxes.

### rename.py

Utility script used to rename annotation files downloaded from Roboflow so that filenames match the project's dataset organization.

---

## 4. pkl_files_generated/

Contains the trained regression models generated after executing:

```
train_model_width_weight.py
```

Generated files:

```
weight_features_D.pkl
weight_model_D_huber.pkl
width_features.pkl
width_model.pkl
```

These models are loaded during real-time deployment:

```python
width_model = joblib.load("pkl_files_generated/width_model.pkl")
width_features = joblib.load("pkl_files_generated/width_features.pkl")

weight_model = joblib.load("pkl_files_generated/weight_model_D_huber.pkl")
weight_features = joblib.load("pkl_files_generated/weight_features_D.pkl")
```

### Description

**width_model.pkl**

Trained regression model used to estimate the hidden width of a detected book.

**width_features.pkl**

Stores the selected feature configuration used during width model training to ensure identical feature ordering during deployment.

**weight_model_D_huber.pkl**

Final Huber Regression model (Model D) used to estimate book weight from visual and geometric features.

**weight_features_D.pkl**

Stores the selected feature configuration required by the deployed weight prediction model.

---

## 5. subfeatures/

Contains supporting utilities developed throughout the project.

These scripts assist with visualisation, validation, debugging, hardware testing, and experimental analysis.

Examples include:

- plot4.3
- plot4.4.3
- normal_distribution
- depth_measure
- visual_check
- testd435
- autoclicker

These utilities were mainly used to:

- Generate thesis figures
- Visualise regression performance
- Inspect captured datasets
- Verify camera measurements
- Test Intel RealSense D435 functionality
- Automate repetitive experimental tasks

---

## 6. others_nonused_codes/

Contains previous experimental scripts developed during the research and development process.

Examples include:

- regression_step1_vision
- regression_step2AB_kmeans
- regression_step5AB_pixelanddepth
- train_width
- train_weight
- train_width_weight
- boundingbox
- predictwidth
- predictwidthhistogram
- predictwidthClandPWT
- calibrated Dvalue
- checkfixed

These scripts document the project's development history, including prototype implementations, algorithm evaluations, and experimental approaches.

Although they are no longer required for the final deployment pipeline, they are retained for future reference and reproducibility.

---

# Dependencies

Major libraries used:

* Python 3.10.11
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
