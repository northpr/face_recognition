import cv2
import torch
import numpy as np
from classes_dict import data_dict
from face_recog_network import FaceRecog


# Load the state_dict of the trained model
state_dict = torch.load('Face_Recognition_checkpoint.pth', map_location="cpu")
CLASSES=len(data_dict)
model = FaceRecog(num_classes=CLASSES)

# Load the state_dict into the model and prepare for eval mode
model.load_state_dict(state_dict)
model.eval()

# Initialize the webcam and call model to detect face
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    # Read video from webcam by frame and detect the face from the video
    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(rgb_frame, 1.3, 5)

    # Draw a rectangle around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = rgb_frame[y:y + h, x:x + w]

        # Resize the face to (224, 224)
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

        # Draw the predicted class on the frame
        cv2.putText(frame, on_screen_display, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('ResNet34', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()