# Not resized version

import cv2
import pickle
import face_recognition
import numpy as np


# Load model from Pickle file
with open("encodings.pickle", "rb") as f:
    model = pickle.load(f)

# Initialize front video
cap = cv2.VideoCapture(0)
process_frame = True


while True:
    ret, frame = cap.read()
    rgb_frame = frame[:,:,::-1]

    # Find all fthe faces and encoding
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    
    # Loop through face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        match = face_recognition.compare_faces(model["encodings"], face_encoding)
        name = "Unknown"

        if True in match:
            first_match_index = match.index(True)
            name = model["names"][first_match_index]

        # Draw a box on every face
        cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255),2)
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom-35), (right, bottom),(0,255,0), cv2.FILLED)
        cv2.putText(frame, name, (left+6, bottom-6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,0),2)

        cv2.imshow("test", frame)

            
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()

cv2.destroyAllWindows()
    