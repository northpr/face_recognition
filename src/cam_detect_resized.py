# Resize version to have better webcam performance

import cv2
import pickle
import face_recognition
import numpy as np


# Load model from Pickle file
with open("encodings.pickle", "rb") as f:
    model = pickle.load(f)

# Initialize front video
cap = cv2.VideoCapture(0)

# Initialize variables
face_locations = []
face_encodings = []
face_names = []
process_frame = True

while True:
    _, frame = cap.read()
    if process_frame:
        small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)

        # Convert the image to RGB
        rgb_small_frame = small_frame[:,:,::-1]
        

        # Find all fthe faces and encoding
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        # Loop through face
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(model["encodings"], face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = model["names"][first_match_index]


            # face_distances = face_recognition.face_distance(model["encodings"], face_encoding)
            # best_match_index = np.argmin(face_distances)
            # if matches[best_match_index]:
            #     name = model["names"]
            face_names.append(name)

    process_frame = not process_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4


        # Draw a box on every face
        cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255),2)
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom-35), (right, bottom),(0,255,0), cv2.FILLED)
        cv2.putText(frame, str(name), (left+6, bottom-6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,0),2)

    cv2.imshow("face_recognition", frame)

            
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
    