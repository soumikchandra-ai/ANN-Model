# 🔢 Handwritten Digit Classification using ANN

## 📌 Overview
This project implements an **Artificial Neural Network (ANN)** to classify handwritten digits (0–9). The model learns patterns from pixel values of images and predicts the correct digit. It is a beginner-friendly project that demonstrates the fundamentals of neural networks and supervised learning.

---

## 🎯 Objectives
- Build an ANN for handwritten digit classification
- Train the model on image data
- Evaluate performance using standard metrics
- Understand forward and backward propagation in neural networks

---

## 🧠 Model Architecture
- **Input Layer:** 784 neurons (28×28 flattened image)
- **Hidden Layers:** Dense layers with ReLU activation
- **Output Layer:** 10 neurons with Softmax activation

---

## 📂 Dataset
- **Name:** MNIST Handwritten Digits Dataset
- **Classes:** Digits from 0 to 9
- **Image Size:** 28 × 28 grayscale images

---

## ⚙️ Tech Stack
- Python
- NumPy
- Pandas
- Matplotlib
- TensorFlow

---

## 🛠️ Implementation Steps
1. Load and preprocess the MNIST dataset
2. Normalize pixel values
3. Define the ANN architecture
4. Train the model using backpropagation
5. Evaluate accuracy on test data
6. Visualize predictions

---

## 📊 Evaluation Metrics
- Accuracy
- Loss
- Confusion Matrix
- Classification Report

---

## 🚀 Results
- Achieved high accuracy on test data
- Successfully classified handwritten digits

---

## ▶️ How to Run
```bash
git clone https://github.com/your-username/ann-digit-classification.git
cd ann-digit-classification
pip install -r requirements.txt
python train.py
