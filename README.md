# YOLO Model Deployment on Local Device

This project demonstrates how to deploy and run a trained YOLO model locally. The included `yolo_detect.py` script shows how to load a model, run inference on a video/image source (e.g. webcam), and display detection results.

---

## 🛠 Requirements

- Python 3.12 (you can choose the version you are using)
- [Anaconda](https://www.anaconda.com/download)
- Optional: NVIDIA GPU with CUDA support

---

## 🚀 Setup Instructions

### 1. Install Anaconda

Download Anaconda for your OS from [here](https://www.anaconda.com/download), skip registration, and install with default options.

### 2. Create and activate a virtual environment

Open **Anaconda Prompt** (Windows) or your terminal (Linux/macOS), then run:

```bash
conda create --name yolo-env1 python=3.12 -y
conda activate yolo-env1
```

### 3. Install required packages

In **Anaconda Prompt**, run:

```bash
pip install ultralytics
```

> ✅ For NVIDIA GPU acceleration (CUDA 12.8), install PyTorch with CUDA:

```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

---

## 📦 Model Preparation

1. Unzip the repo to a folder on your PC.
2. Change directory of the Anaconda prompt to that folder:

```bash
cd path/to/my_model_folder
```

---

## 🎯 Run Inference

Run real-time object detection on your USB webcam (1280x720 resolution):

```bash
python yolo_detect.py --model my_model.pt --source usb0 --resolution 1280x720
```

Run real-time object detection on your own images/videos (Example: 05_spur_07.jpg):

\\ Get the images from "test_images" to the folder containing your model

```bash
python yolo_detect.py --model my_model.pt --source 05_spur_07.jpg --save_image
```

---

## 📺 Output

A window will appear showing the live webcam/images/videos feed with bounding boxes around detected objects
Image inference will be saved in the same folder

---

## 📹 Tutorial

tutorial link: https://www.youtube.com/watch?v=r0RspiLG260&t=983s

---

