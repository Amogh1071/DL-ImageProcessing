import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np


def get_image_embeddings(image_paths, model_path='embedder.tflite'):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ImageEmbedderOptions(
        base_options=base_options,
        l2_normalize=True,
        quantize=False
    )

    embeddings = {}

    with vision.ImageEmbedder.create_from_options(options) as embedder:
        for path in image_paths:
            try:
                mp_image = mp.Image.create_from_file(path)


                cv_image = cv2.imread(path)
                if cv_image is None:
                    raise ValueError(f"Could not read image at {path}")

                embedding_result = embedder.embed(mp_image)
                feature_vector = embedding_result.embeddings[0].embedding
                embeddings[path] = feature_vector


                text = f"Embedding: {np.array2string(feature_vector[:5], precision=2, separator=', ')}..."

                # Resize image for display if too large
                height, width = cv_image.shape[:2]
                if height > 800 or width > 800:
                    cv_image = cv2.resize(cv_image, (int(width * 0.5), int(height * 0.5)))


                cv2.putText(cv_image, text,
                            (10, 30),  # Position
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,  # Font scale
                            (0, 255, 0),  # Green color
                            1)  # Thickness

                # Show image
                cv2.imshow('Image with Embedding', cv_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            except Exception as e:
                print(f"Error processing {path}: {str(e)}")

    return embeddings


if __name__ == "__main__":

    image_paths = [
        r'C:\Users\Asus\PycharmProjects\JupyterProject\data\s2.jpg'
    ]

    embeddings = get_image_embeddings(image_paths)


    for path, vector in embeddings.items():
        print(f"\nEmbedding for {path}:")
        print(f"Vector shape: {len(vector)} dimensions")
        print(f"First 5 values: {vector[:5]}")
        print(f"Array type: {type(vector)}")