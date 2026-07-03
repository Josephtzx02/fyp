# YOLOv8n-OBB Training Setup

This folder contains the required files and instructions to reproduce the custom **YOLOv8 Nano Oriented Bounding Box (YOLOv8n-OBB)** training used in this project.

The model was trained to detect **book spines** using oriented bounding boxes (OBB), enabling accurate estimation of book orientation for the RGB-D robotic vision pipeline.

---

# Prerequisites

The training environment was developed and tested using:

| Component   | Version                    |
| ----------- | -------------------------- |
| Python      | 3.10.11                    |
| CUDA        | 12.1                       |
| PyTorch     | CUDA 12.1 build            |
| Ultralytics | Latest compatible version  |
| GPU         | NVIDIA RTX 4050 Laptop GPU |

---

# Files

| File             | Description                                                                                                        |
| ---------------- | ------------------------------------------------------------------------------------------------------------------ |
| `data.yaml`      | Dataset configuration file containing the training, validation and testing dataset paths.                          |
| `yolov8n-obb.pt` | Official pretrained YOLOv8 Nano Oriented Bounding Box model used as the starting checkpoint for transfer learning. |

---

# Step 1 — Create a Python Virtual Environment

It is recommended to use a dedicated virtual environment.

```bash
py -3.10 -m venv yolov8_gpu
```

Activate the environment:

```bash
yolov8_gpu\Scripts\activate
```

---

# Step 2 — Install PyTorch (CUDA 12.1)

Install the CUDA-enabled version of PyTorch.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

# Step 3 — Install Required Libraries

Install all required Python packages.

```bash
pip install ultralytics
```

```bash
pip install opencv-python numpy pandas scikit-learn pyrealsense2 pycocotools rdp matplotlib joblib
```

---

# Step 4 — Verify Installation

Check that Ultralytics has been installed correctly.

```bash
pip show ultralytics
```

---

# Step 5 — Prepare the Dataset

Before training, ensure that:

* the dataset has been annotated
* polygon annotations have been converted into YOLOv8-OBB labels
* the dataset directory matches the paths defined inside `data.yaml`

The preprocessing pipeline used in this project consists of:

1. Roboflow annotation
2. Polygon simplification using the Ramer-Douglas-Peucker (RDP) algorithm
3. Convex Hull generation
4. Minimum-area oriented bounding box fitting
5. Conversion to YOLOv8-OBB label format

Please refer to the **datafilter/** folder for the preprocessing scripts.

---

# Step 6 — Start Training

Run the following command to begin training.

```bash
yolo task=obb mode=train model=yolov8n-obb.pt data=data.yaml epochs=200 imgsz=640 batch=8
```

### Training Parameters

| Parameter              | Description                                                                  |
| ---------------------- | ---------------------------------------------------------------------------- |
| `task=obb`             | Enables Oriented Bounding Box training instead of standard object detection. |
| `mode=train`           | Starts the training process.                                                 |
| `model=yolov8n-obb.pt` | Uses the pretrained YOLOv8 Nano OBB model for transfer learning.             |
| `data=data.yaml`       | Specifies the dataset configuration file.                                    |
| `epochs=200`           | Maximum number of training epochs.                                           |
| `imgsz=640`            | Resizes input images to 640 × 640 pixels.                                    |
| `batch=8`              | Batch size used during training.                                             |

The trained weights will be saved automatically under:

```text
runs/
└── obb/
    └── train/
        └── weights/
            ├── best.pt
            └── last.pt
```

---

# Resume Training (Optional)

If training is interrupted, it can be resumed from the latest checkpoint.

```bash
yolo train resume model=runs/obb/train/weights/last.pt
```

---

# Predict Using the Trained Model

Run inference on the testing dataset.

```bash
yolo task=obb mode=predict model=runs/obb/train/weights/best.pt source=book_obb_dataset_v1/images/test imgsz=640 conf=0.75
```

### Parameters

| Parameter | Description                                |
| --------- | ------------------------------------------ |
| `model`   | Trained YOLOv8n-OBB model.                 |
| `source`  | Folder containing images for inference.    |
| `imgsz`   | Input image resolution.                    |
| `conf`    | Confidence threshold for object detection. |

Prediction results will be saved automatically inside the `runs/obb/predict/` directory.

---

# Benchmark Inference Speed (Optional)

Benchmark the trained model on the CPU.

```bash
yolo benchmark model=best.pt imgsz=640 device=cpu
```

Benchmark the trained model on the GPU.

```bash
yolo benchmark model=best.pt imgsz=640 device=0
```

`device=0` refers to the first available CUDA GPU (NVIDIA RTX 4050 in this project).

The benchmark reports inference speed, preprocessing time, postprocessing time, and throughput, allowing comparison between CPU and GPU performance.

---

# Output

After successful training, the most important output files are:

| File      | Description                                                                                |
| --------- | ------------------------------------------------------------------------------------------ |
| `best.pt` | Best-performing model based on validation metrics. Recommended for deployment.             |
| `last.pt` | Final checkpoint from the most recent training epoch. Used to resume interrupted training. |

The `best.pt` model produced during this stage is subsequently used by the real-time deployment pipeline (`obb_clickable_demo.py`) for book spine detection and orientation estimation.

---

# Notes

* This repository uses **YOLOv8 Oriented Bounding Box (OBB)** rather than standard YOLO object detection (`task=detect`).
* The pretrained `yolov8n-obb.pt` model serves as the transfer learning backbone.
* Dataset preprocessing is a critical step before training and should be completed using the scripts provided in the `datafilter/` folder.
* Training performance depends on GPU capability. The experiments reported in this project were performed using an NVIDIA RTX 4050 Laptop GPU.
