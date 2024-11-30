import os
import cv2

# Directory where the dataset will be stored
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Number of classes to collect data
number_of_classes = 3
dataset_size = 100

# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the camera")
    exit()

# Loop through each class
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    # Wait for user to press "Q" to start collecting images
    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("Error: Failed to capture frame")
            continue
    
        cv2.putText(frame, 'Press "Q" to collect data!', (100, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        
        # Wait for 'Q' key press to start capturing images
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Capture the specified number of samples and save them
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("Error: Failed to capture frame, skipping")
            continue

        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        
        flipped_frame = cv2.flip(frame, 1)

        cv2.imwrite(os.path.join(class_dir, f'{dataset_size + counter}.jpg'), flipped_frame)

        counter += 1

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
