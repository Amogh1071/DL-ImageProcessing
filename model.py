import numpy as np
import cv2
import mediapipe as mp
from keras.models import load_model

model = load_model('model.h5')
label = np.load('labels.npy')

# Initialize MediaPipe models
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
mp_hands = mp.solutions.hands
drawing_utils = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break
    frame_bgr = cv2.flip(frame_bgr, 1)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Process frame with holistic model
    results = holistic.process(frame_rgb)

    lst = []

    # Extract face landmarks (468 landmarks, 2 coordinates each = 936 features)
    if results.face_landmarks:
        ref_x = results.face_landmarks.landmark[1].x
        ref_y = results.face_landmarks.landmark[1].y
        for landmark in results.face_landmarks.landmark:
            lst.append(landmark.x - ref_x)
            lst.append(landmark.y - ref_y)
    else:
        # Pad zeros if face not detected
        lst.extend([0.0] * 468 * 2)  # 936 zeros

    # Extract left hand landmarks (21 landmarks, 2 coordinates each = 42 features)
    if results.left_hand_landmarks:
        ref_x = results.left_hand_landmarks.landmark[8].x
        ref_y = results.left_hand_landmarks.landmark[8].y
        for landmark in results.left_hand_landmarks.landmark:
            lst.append(landmark.x - ref_x)
            lst.append(landmark.y - ref_y)
    else:
        # Pad zeros if left hand not detected
        lst.extend([0.0] * 21 * 2)  # 42 zeros

    # Extract right hand landmarks (21 landmarks, 2 coordinates each = 42 features)
    if results.right_hand_landmarks:
        ref_x = results.right_hand_landmarks.landmark[8].x
        ref_y = results.right_hand_landmarks.landmark[8].y
        for landmark in results.right_hand_landmarks.landmark:
            lst.append(landmark.x - ref_x)
            lst.append(landmark.y - ref_y)
    else:
        # Pad zeros if right hand not detected
        lst.extend([0.0] * 21 * 2)  # 42 zeros

    # Ensure exactly 1020 features (936 + 42 + 42)
    assert len(lst) == 1020, f"Expected 1020 features, got {len(lst)}"

    # Reshape for model input (batch_size=1, features=1020)
    input_data = np.array(lst).reshape(1, -1)

    # Predict
    pred = np.argmax(model.predict(input_data))
    predicted_label = label[pred]

    # Display prediction
    cv2.putText(
        frame_bgr,
        str(predicted_label),
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Draw landmarks
    if results.face_landmarks:
        drawing_utils.draw_landmarks(
            frame_bgr,
            results.face_landmarks,
            mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=drawing_utils.DrawingSpec(color=(0, 255, 0), circle_radius=1)
        )
    if results.left_hand_landmarks:
        drawing_utils.draw_landmarks(
            frame_bgr,
            results.left_hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )
    if results.right_hand_landmarks:
        drawing_utils.draw_landmarks(
            frame_bgr,
            results.right_hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

    cv2.imshow('frame', frame_bgr)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()