import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Paths
DATASET_PATH = "C:/Users/Asus/PycharmProjects/JupyterProject/fer2013/train"  # Change this to your dataset path
MODEL_PATH = "embedder.tflite"  # Ensure you have the correct model file

# Load MediaPipe Image Embedder
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.ImageEmbedderOptions(base_options=base_options, l2_normalize=True, quantize=False)

# Initialize lists for DataFrame
image_filenames = []
expressions = []
embeddings_list = []

# Process images
with vision.ImageEmbedder.create_from_options(options) as embedder:
    for expression in os.listdir(DATASET_PATH):  # Loop through emotion folders
        expression_path = os.path.join(DATASET_PATH, expression)

        if not os.path.isdir(expression_path):
            continue  # Skip non-folder items

        print(f"Processing: {expression}")

        for img_file in tqdm(os.listdir(expression_path)):  # Process each image
            img_path = os.path.join(expression_path, img_file)

            try:
                # Load image with OpenCV
                cv_image = cv2.imread(img_path)

                # Convert grayscale to RGB (MediaPipe expects RGB images)
                if cv_image is None:
                    print(f"Skipping {img_path} (could not read)")
                    continue

                if len(cv_image.shape) == 2 or cv_image.shape[2] == 1:
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)

                # Convert OpenCV image to MediaPipe image format
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_image)

                # Get embedding
                embedding_result = embedder.embed(mp_image)
                feature_vector = embedding_result.embeddings[0].embedding  # 1280-dimensional vector

                # Store results
                image_filenames.append(img_file)
                expressions.append(expression)
                embeddings_list.append(feature_vector)

            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")

# Convert data to Pandas DataFrame
df = pd.DataFrame(embeddings_list)
df.insert(0, "Image", image_filenames)
df.insert(1, "Expression", expressions)

# Save to Excel
EXCEL_PATH = "fer2013_embeddings.xlsx"
df.to_excel(EXCEL_PATH, index=False)

print(f"Saved embeddings to {EXCEL_PATH}")
