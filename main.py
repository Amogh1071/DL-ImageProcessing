import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

name = input("Enter name of data: ")

cap = cv2.VideoCapture(0)
faceMesh = mp.solutions.face_mesh
face = faceMesh.FaceMesh(static_image_mode=False, max_num_faces=2, min_detection_confidence=0.6, min_tracking_confidence=0.6 )
draw = mp.solutions.drawing_utils

holistic =mp.solutions.holistic
holis = holistic.Holistic()
hands = mp.solutions.hands

X =[]
data_size=0

while True:
    ret, frame_bgr = cap.read()

    lst = []
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    results = face.process(frame_rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            draw.draw_landmarks(frame_bgr, face_landmarks, faceMesh.FACEMESH_CONTOURS, landmark_drawing_spec = draw.DrawingSpec(color=(0, 255, 0), circle_radius=1))

    res = holis.process(frame_rgb)

    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)

    if res.left_hand_landmarks and res.left_hand_landmarks.landmark:
        reference_point = res.left_hand_landmarks.landmark[8]  # Ensure index 8 exists
        for i in res.left_hand_landmarks.landmark:
            lst.append(i.x - reference_point.x)
            lst.append(i.y - reference_point.y)
    else:
        lst.extend([0.0] * 42)  # Fill missing values with zeros

    if res.right_hand_landmarks and res.right_hand_landmarks.landmark:
        reference_point = res.right_hand_landmarks.landmark[8]  # Ensure index 8 exists
        for i in res.right_hand_landmarks.landmark:
            lst.append(i.x - reference_point.x)
            lst.append(i.y - reference_point.y)
    else:
        lst.extend([0.0] * 42)  # Fill missing values with zeros

    X.append(lst)

    data_size+=1


    draw.draw_landmarks(frame_bgr, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    draw.draw_landmarks(frame_bgr, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

    cv2.putText(frame_bgr,str(data_size), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('frame', frame_bgr)

    if cv2.waitKey(1) == 27 or data_size>199:
        cap.release()
        cv2.destroyAllWindows()
        break

np.save(f"{name}.npy", np.array(X) )

