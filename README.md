# 📈 SmartPattern: Stock Chart Pattern Detection using PyTorch

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

SmartPattern is an end-to-end deep learning system that detects **20 technical chart patterns** from candlestick stock chart images. It uses a **ResNet18-based multi-label CNN** trained with PyTorch, and exposes a real-time prediction interface via a **Streamlit web app**.

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [Architecture](#-architecture)
3. [Data Flow](#-data-flow)
4. [Pipeline: Step-by-Step File Execution](#-pipeline-step-by-step-file-execution)
5. [Project Structure](#-project-structure)
6. [Detected Patterns (20 Classes)](#-detected-patterns-20-classes)
7. [Installation](#-installation)
8. [Usage](#-usage)
9. [Model Details](#-model-details)
10. [License](#-license)
11. [Author](#️-author)

---

## 🔭 Project Overview

SmartPattern automates the identification of technical chart patterns that traders use for market analysis. Given a candlestick chart image, the model predicts which of the **20 chart patterns** are present — supporting **multi-label classification** (multiple patterns can co-exist in one chart).

**Key Highlights:**
- Multi-label classification over 20 technical chart patterns
- ResNet18 backbone fine-tuned with BCEWithLogitsLoss and class-balanced weights
- Streamlit web UI for instant, interactive predictions
- Full preprocessing pipeline from raw YOLO-format labels to one-hot encoded CSV

---

## 🏗 Architecture

```
Raw Chart Images  ──►  Preprocessing  ──►  PyTorch Dataset  ──►  ResNet18 CNN  ──►  Multi-label Output
   (JPG/PNG)           (clean + rename)     (augment + load)    (fine-tuned)       (20 binary classes)
                                                                      │
                                                          models/chart_pattern_model.pth
                                                                      │
                                                           Streamlit Web App (inference)
```

**Model layers:**

```
Input (224×224×3)
    │
ResNet18 Backbone (pretrained ImageNet weights)
    │   ├── Conv1 → BN → ReLU → MaxPool
    │   ├── Layer1 (2× BasicBlock)
    │   ├── Layer2 (2× BasicBlock, stride=2)
    │   ├── Layer3 (2× BasicBlock, stride=2)
    │   └── Layer4 (2× BasicBlock, stride=2)
    │
AdaptiveAvgPool2d → Flatten
    │
Linear(512 → 20)   ← replaced final FC layer
    │
Output logits (20 classes) — apply Sigmoid for probabilities
```

---

## 🔄 Data Flow

```
data/raw/train/
  ├── images/         ← Raw chart images (JPG/PNG, mixed filenames)
  └── labels/         ← YOLO-format .txt files (class_id per line)
          │
          ▼
  [Step 1] data_preprocessing.py
          Remove images with empty label files
          │
          ▼
  [Step 2] image_preprocessing.py
          Rename images sequentially (1.jpg, 2.jpg, …)
          Copy to data/processed/train_images/
          │
          ▼
  [Step 3] label_preprocessing.py
          Parse YOLO labels → one-hot vectors (20 classes)
          Output: data/processed/train_labels.csv
          │
          ▼
  [Step 4] chart_dataset.py
          ChartPatternDataset (PyTorch Dataset)
          Augmentation: Resize → Flip → Rotate → ColorJitter → Normalize
          DataLoader (batch_size=32)
          │
          ▼
  [Step 5] model.py
          ChartPatternCNN (ResNet18 backbone, FC→20)
          │
          ▼
  [Step 6] train.py
          BCEWithLogitsLoss + pos_weight class balancing
          Adam optimizer (lr=0.001) + StepLR scheduler
          25 epochs → saves models/chart_pattern_model.pth
          │
          ▼
  [Step 7] streamlit_app/app.py
          Load saved model → accept uploaded image
          Predict top-3 patterns with confidence scores
          Display bar chart of predictions
```

---

## 🚀 Pipeline: Step-by-Step File Execution

### Step 1 — `utils/data_preprocessing.py`
**Purpose:** Clean the raw dataset by removing samples with empty or missing label files.

- Iterates over all raw images in `data/raw/train/images/`
- Reads the corresponding `.txt` label file from `data/raw/train/labels/`
- Deletes both the image **and** the label if the label file is empty or absent
- Prints a summary of how many pairs were removed

```bash
python utils/data_preprocessing.py
```

---

### Step 2 — `utils/image_preprocessing.py`
**Purpose:** Standardize image filenames by renaming them to sequential integers.

- Reads all valid images from `data/raw/train/images/`
- Renames them as `1.jpg`, `2.jpg`, `3.jpg`, … in sorted order
- Copies renamed images to `data/processed/train_images/`

```bash
python utils/image_preprocessing.py
```

---

### Step 3 — `utils/label_preprocessing.py`
**Purpose:** Convert YOLO-format text labels into a one-hot encoded CSV.

- Pairs each renamed image with its YOLO label file
- Reads class IDs from each `.txt` file and builds a 20-dimensional binary vector
- Saves the result as `data/processed/train_labels.csv` with columns:
  `Filename, Class0, Class1, …, Class19`
- Prints the per-class label distribution

```bash
python utils/label_preprocessing.py
```

---

### Step 4 — `utils/chart_dataset.py`
**Purpose:** Define the PyTorch `Dataset` and `DataLoader` used during training.

- `ChartPatternDataset` reads images from `data/processed/train_images/` and labels from `data/processed/train_labels.csv`
- Applies the following augmentation pipeline at load time:
  - Resize to 224×224
  - Random horizontal flip (p=0.5)
  - Random rotation (±5°)
  - Color jitter (brightness/contrast ±10%)
  - Normalize (mean=0.5, std=0.5 per channel)
- Wraps the dataset in a `DataLoader` with `batch_size=32` and `shuffle=True`

> This file is imported by `train.py`; it does not need to be run directly.

---

### Step 5 — `utils/model.py`
**Purpose:** Define the CNN model architecture.

- Loads `ResNet18` with pretrained ImageNet weights
- Replaces the final fully-connected layer with `Linear(512, 20)` for multi-label output
- Provides `save_model()` and `load_model()` helper methods

> This file is imported by both `train.py` and the Streamlit app; it does not need to be run directly.

---

### Step 6 — `utils/train.py`
**Purpose:** Train the model end-to-end and save the weights.

- Reads `data/processed/train_labels.csv` to compute per-class positive weights for `BCEWithLogitsLoss`
- Initializes `ChartPatternCNN` and moves it to GPU (falls back to CPU)
- Trains for **25 epochs** with:
  - **Optimizer:** Adam (lr=0.001)
  - **Scheduler:** StepLR — halves lr every 10 epochs
  - **Loss:** BCEWithLogitsLoss with `pos_weight` for class imbalance
- Reports epoch-level loss and exact-match multi-label accuracy
- Saves the final model to `models/chart_pattern_model.pth`

```bash
cd utils
python train.py
```

---

### Step 7 — `streamlit_app/app.py`
**Purpose:** Serve real-time predictions via an interactive web interface.

- Loads the trained model from `models/chart_pattern_model.pth`
- Accepts a user-uploaded chart image (JPG/PNG)
- Applies the same resize + normalize transform used in training
- Runs inference and displays the **top 3 predicted patterns** with confidence scores
- Renders an interactive bar chart of predicted probabilities

```bash
streamlit run streamlit_app/app.py
```

---

## 📁 Project Structure

```
SmartPattern/
│
├── data/
│   ├── raw/
│   │   └── train/
│   │       ├── images/          ← Raw chart images
│   │       └── labels/          ← YOLO-format .txt label files
│   └── processed/
│       ├── train_images/        ← Cleaned & renamed images (generated)
│       └── train_labels.csv     ← One-hot encoded label matrix (generated)
│
├── utils/
│   ├── __init__.py
│   ├── data_preprocessing.py    ← Step 1: Clean empty labels & images
│   ├── image_preprocessing.py   ← Step 2: Rename images sequentially
│   ├── label_preprocessing.py   ← Step 3: Generate train_labels.csv
│   ├── chart_dataset.py         ← Step 4: PyTorch Dataset & DataLoader
│   ├── model.py                 ← Step 5: ResNet18 CNN model definition
│   └── train.py                 ← Step 6: Training loop & model saving
│
├── models/
│   └── chart_pattern_model.pth  ← Saved model weights (generated after training)
│
├── streamlit_app/
│   └── app.py                   ← Step 7: Streamlit inference web app
│
├── notebooks/
│   └── cnn_model.py             ← Experimental notebook scripts
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🗂 Detected Patterns (20 Classes)

| Class ID | Pattern Name             | Class ID | Pattern Name              |
|----------|--------------------------|----------|---------------------------|
| 0        | Ascending Triangle       | 10       | Resistance Emerging       |
| 1        | Channel Down             | 11       | Resistance Breakout       |
| 2        | Channel Up               | 12       | Rising Wedge              |
| 3        | Cup and Handle           | 13       | Rounding Bottom           |
| 4        | Descending Triangle      | 14       | Rounding Top              |
| 5        | Double Bottom            | 15       | Support Breakout          |
| 6        | Double Top               | 16       | Triangle                  |
| 7        | Falling Wedge            | 17       | Triple Bottom             |
| 8        | Head and Shoulders       | 18       | Triple Top                |
| 9        | Inverse Head & Shoulders | 19       | Rectangle                 |

---

## ⚙️ Installation

**1. Clone the repository:**

```bash
git clone https://github.com/soham29640/SmartPattern_Stock_Chart_Pattern_Detection_using_PyTorch.git
cd SmartPattern_Stock_Chart_Pattern_Detection_using_PyTorch
```

**2. Create and activate a virtual environment (recommended):**

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## 🖥 Usage

### Training from Scratch

Run the preprocessing steps in order, then train:

```bash
# Step 1 – Clean raw data
python utils/data_preprocessing.py

# Step 2 – Rename and copy images
python utils/image_preprocessing.py

# Step 3 – Generate label CSV
python utils/label_preprocessing.py

# Step 4–6 – Train the model (imports dataset and model internally)
cd utils
python train.py
```

The trained model will be saved to `models/chart_pattern_model.pth`.

### Running the Web App

```bash
streamlit run streamlit_app/app.py
```

Open the URL shown in the terminal (default: `http://localhost:8501`), upload a candlestick chart image via the sidebar, and view the top-3 predicted patterns with confidence scores.

---

## 🤖 Model Details

| Property          | Value                           |
|-------------------|---------------------------------|
| Backbone          | ResNet18 (pretrained ImageNet)  |
| Input Size        | 224 × 224 × 3                   |
| Output            | 20 logits (multi-label)         |
| Loss Function     | BCEWithLogitsLoss + pos_weight  |
| Optimizer         | Adam (lr=0.001)                 |
| LR Scheduler      | StepLR (step=10, γ=0.5)         |
| Epochs            | 25                              |
| Batch Size        | 32                              |
| Evaluation Metric | Exact multi-label match accuracy|
| Hardware          | GPU (CUDA) / CPU fallback       |

---

## 📝 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙋‍♂️ Author

**Soham Samanta**  
AI/ML Enthusiast | Deep Learning Practitioner

[![GitHub](https://img.shields.io/badge/GitHub-soham29640-181717?logo=github)](https://github.com/soham29640)
