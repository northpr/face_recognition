import cv2
import torch
import numpy as np
from classes_dict import data_dict
from face_recog_network import FaceRecog
from collections import Counter
import time

# Load all paramemters and state_dict of the trained model
state_dict = torch.load('Face_Recognition_checkpoint.pth', map_location="cpu")
model = FaceRecog(num_classes=len(data_dict))
model.load_state_dict(state_dict)
model.eval()

# Initialize list to store idx values and for itme read

# Initialize the webcam and call model to detect face
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    # Read video from webcam by frame and detect the face from the video
    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(rgb_frame, 1.3, 5)

    # Check if only one face is detected
    if len(faces) == 1:
        pred_list = []
        # Draw a rectangle around each detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = rgb_frame[y:y + h, x:x + w]

            # Manipulate face Resize the face to (224, 224)
            face = cv2.resize(face, (224, 224))
            face = torch.from_numpy(face).float()
            face = face.permute(2,0,1)

            # Normalize the face tensor and predict by using our model
            face = (face - face.mean()) / face.std()
            output = model(face.unsqueeze(0))

            # Get the predicted class
            _, pred = torch.max(output, 1)
            idx = pred.item()
            on_screen_display = f"{idx} - {data_dict[idx]['full_name']}"
            cv2.rectangle(frame, (x,y-50), (x+300, y-10), (0,255,0), -1)
            cv2.putText(frame, on_screen_display, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
            # time.sleep(3)
            # else:
            #     most_common_idx = max(set(pred_list), key=pred_list.count)
            #     on_screen_display = f"{idx} - {data_dict[most_common_idx]['full_name']}"
            #     cv2.putText(frame, on_screen_display, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    else:
        cv2.putText(frame, "Please show only one face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow('ResNet34', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()