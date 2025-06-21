# 📈 SmartPattern: Stock Chart Pattern Detection using PyTorch

This project implements a multi-label classification system to detect technical chart patterns from stock images using deep learning (CNN with ResNet18). It automates the recognition of multiple patterns in a single chart, aiding traders and analysts in decision-making.

---

## 🔧 Features & Efforts

- **Chart Image Preprocessing**:
  - Cleaned and filtered raw images and labels
  - Handled missing and duplicate labels
  - Converted filenames to numerical format for consistency
  - Generated one-hot encoded label matrix for multi-label classification

- **Dataset Construction**:
  - Designed custom `ChartPatternDataset` in PyTorch
  - Applied advanced data augmentations (rotation, flip, color jitter)
  - Supported multi-label loading with dynamic class mapping

- **Model Architecture**:
  - Built a deep CNN using **ResNet18 backbone**
  - Modified the final layer for **multi-label outputs (20 classes)**

- **Training Strategy**:
  - Used **BCEWithLogitsLoss** with **class balancing using `pos_weight`**
  - Optimized with **Adam**, and included **learning rate scheduling**
  - Ensured robust multi-GPU training compatibility

- **Evaluation**:
  - Accuracy calculated using strict multi-label match
  - Trained on GPU (with CPU fallback support)

---

## 📁 Project Structure

data/
  ├── raw/
  └── processed/
utils/
  ├── chart_dataset.py
  └── train.py
models/
train_labels.csv
README.md
.gitignore

---

## 📝 License

This project is licensed under the **MIT License**.

---

## 🙋‍♂️ Author

**Soham Samanta**  
AI/ML Enthusiast | Deep Learning Practitioner
