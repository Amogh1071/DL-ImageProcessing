# Facial Expression Embeddings with MediaPipe (FER2013)

## 📌 Overview
This project involves the extraction of facial expression embeddings using **MediaPipe** from the **FER2013 dataset** and training a classifier to recognize various facial expressions. The embeddings are saved to an Excel file for further processing.

---

## 📂 Dataset
The **FER2013 dataset** consists of grayscale images representing facial expressions, classified into seven categories:
- 😃 Happy
- 😢 Sad
- 😠 Angry
- 😨 Fear
- 😐 Neutral
- 😱 Surprise
- 😞 Disgust

The dataset is publicly available on Kaggle and needs to be structured as follows:
```
FER2013/
│
├── train/
├── validation/
└── test/
```

---

## 📝 Features
✅ Loads the **FER2013 dataset** from Kaggle.
✅ Processes grayscale images appropriately.
✅ Extracts **1280-dimensional embeddings** using MediaPipe's Image Embedder.
✅ Saves the results (Image, Expression, Embedding) to an **Excel file**.

---

## 🚀 Embedding Extraction Process
1. Load the **FER2013 dataset**.
2. Convert grayscale images to RGB as MediaPipe works with RGB images.
3. Extract embeddings using MediaPipe's Image Embedder (1280-dimensional).
4. Save embeddings, labels, and image references to an Excel file.

---

## 📊 Classifier Model
A Sequential Model was built using the extracted embeddings to classify facial expressions. The model architecture is as follows:

```
- Input Layer: 4096
- Dense Layer: 512 (ReLU Activation)
- Dropout: 0.5
- Output Layer: 7 (Softmax Activation)
- Total Parameters: 2,489,095
```

---

## 🔍 Evaluation
The model was evaluated using a Confusion Matrix, achieving good accuracy across various classes. 

---

## 📦 Dependencies
- TensorFlow
- MediaPipe
- Pandas
- OpenCV
- XlsxWriter

Install the dependencies via pip:
```
pip install tensorflow mediapipe pandas opencv-python xlsxwriter
```

---

## 📌 Usage
1. Clone the repository.
2. Install dependencies.
3. Run the embedding extraction script to generate the Excel file.
4. Train the classifier using the saved embeddings.

---

## 📧 Contact
For any inquiries, feel free to reach out.
