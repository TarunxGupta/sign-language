import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

# Load trained model
model = tf.keras.models.load_model('gesture_model.h5')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(min_detection_confidence=0.5)

# Labels dictionary
labels_dict = {0: 'Peace', 1: 'Close', 2: 'Ok', 3: 'Unknown'}

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]

            min_x, min_y = min(x_), min(y_)
            max_x, max_y = max(x_), max(y_)

            data_aux = [(x - min_x) / (max_x - min_x) for x in x_] + \
                       [(y - min_y) / (max_y - min_y) for y in y_]

            avg_x = np.mean(x_)
            if avg_x < 0.5:
                hand_label = 'Left Hand'
            else:
                hand_label = 'Right Hand'

            prediction = model.predict(np.array([data_aux]), verbose=0)
            predicted_label = np.argmax(prediction)

            if prediction[0][predicted_label] < 0.6:
                predicted_label = 3

            label_text = labels_dict[predicted_label]

            h, w, _ = frame.shape
            x1, y1 = int(min_x * w), int(min_y * h)
            x2, y2 = int(max_x * w), int(max_y * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            label_x = int((x1 + x2 - label_size[0]) / 2)
            label_y = y1 - 10

            cv2.putText(frame, label_text, (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            hand_label_size = cv2.getTextSize(hand_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            hand_label_x = int((x1 + x2 - hand_label_size[0]) / 2)
            hand_label_y = y2 + 30

            cv2.putText(frame, hand_label, (hand_label_x, hand_label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow('Hand Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
