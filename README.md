# Ultrasound Liver and Thyroid Disease Prediction

AI-based ultrasound disease prediction system using **ResNet-18** and **FastAPI** to classify liver, fatty liver, and thyroid ultrasound images with confidence-based outputs.

---

## 🔍 Project Overview

This project is a deep learning–based medical image analysis system designed to predict diseases from ultrasound images. It supports:
- Liver disease classification
- Fatty liver severity prediction
- Thyroid disease classification

The system uses a **ResNet-18 convolutional neural network** trained with a transfer learning approach and is deployed using a **FastAPI backend** with a lightweight web interface for inference.

---

## 🧠 Model Details

- Architecture: ResNet-18 (pretrained on ImageNet)
- Framework: PyTorch
- Approach: Transfer Learning
- Input Size: 224 × 224 ultrasound images
- Output: Disease class with confidence score

Separate trained models are used for:
- Liver disease
- Fatty liver severity
- Thyroid disease

---

## 🛠 Tech Stack

- Python
- PyTorch
- Torchvision
- FastAPI
- Pillow (PIL)
- HTML / CSS

---

## 📁 Project Structure

```
FATTYLIVERPROJECT/
│
├── app/
│   ├── backend/          # FastAPI backend logic
│   └── __init__.py
│
├── data/                 # Liver dataset (Benign, Malignant, Normal)
├── data_fatty/           # Fatty liver dataset (mild, moderate, severe, normal)
├── data_thyroid/         # Thyroid dataset (Benign, Malignant)
│
├── evaluation/           # Model evaluation scripts/results
├── frontend/             # UI files
├── inference/            # Prediction and inference logic
│
├── models/
│   ├── liver/            # Liver trained models
│   ├── fatty_liver/      # Fatty liver trained models
│   ├── thyroid/          # Thyroid trained models
│   └── test_real/
│
├── utils/                # Helper and utility functions
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ▶️ How to Run the Project

### 1️⃣ Create virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Start the FastAPI server
```bash
uvicorn app.backend.app:app --reload
```

### 4️⃣ Open in browser
```
http://127.0.0.1:8000
```

---

## 📊 Datasets

This project uses multiple ultrasound datasets:
- Liver ultrasound images
- Fatty liver ultrasound images
- Thyroid ultrasound images

⚠️ Due to size limitations, datasets are **not included** in the repository.

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**.  
It is **not a medical diagnostic tool** and should not be used for clinical decisions.

---

## 🚀 Future Improvements

- Add Grad-CAM visual explanations
- Improve confidence calibration
- Combine multi-organ prediction into a single pipeline
- Deploy on cloud platforms

---

## 👨‍💻 Contributors

1) Saipranav Sapare
2) Ravi chandra
- Project developed as an academic deep learning application

