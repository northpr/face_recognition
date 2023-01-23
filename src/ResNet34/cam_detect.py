import cv2
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
from pred_classes import star_classes
print(star_classes[96])
class FaceRecog(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # Use a pretrained model
        self.resnet34 = models.resnet34(True)
        self.features = nn.Sequential(*list(self.resnet34.children())[:-1])
        # Replace last layer
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(self.resnet34.fc.in_features, num_classes))

    def forward(self, x):
        x = self.features(x)
        y = self.classifier(x)
        return y
    
    # def summary(self, input_size):
    #     return summary(self, input_size)


# Load the state_dict of the trained model
state_dict = torch.load('Face_Recognition_checkpoint.pth', map_location="cpu")

# Create a new instance of the model
CLASSES=105
model = FaceRecog(num_classes=CLASSES)

# Load the state_dict into the model
model.load_state_dict(state_dict)

# Make sure the model is in eval mode
model.eval()

# Initialize the webcam
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    # Get a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw a rectangle around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop the face from the frame
        face = gray[y:y + h, x:x + w]

        # Resize the face to (224, 224)
        face = cv2.resize(face, (224, 224))
        face = torch.from_numpy(face).float()
        face = face.permute(2,0,1)

        # Normalize the face tensor
        face = (face - face.mean()) / face.std()

        # Pass the face tensor through the model
        output = model(face.unsqueeze(0))

        # Get the predicted class
        _, pred = torch.max(output, 1)

        # Draw the predicted class on the frame
        cv2.putText(frame, star_classes[pred], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('ResNet34', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()