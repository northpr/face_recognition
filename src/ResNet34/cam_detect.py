import cv2
import pickle

with open("Face_Recognition_model.pkl", "rb") as f:
    model = pickle.load(f)

model.eval()

# Load the cascade classifier for face detection - detect the face before label by using the model
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

print("hello world")