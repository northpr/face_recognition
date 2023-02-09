import cv2
import os

def face_extractor(video_path: str, extract_path=None):
    video_file = video_path

    if extract_path is None:
        extract_dir = video_path+ "/images"
    else:
        extract_dir = extract_path

    #Load the video and face_detection
    cap = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Define the face count and face number for the output
    face_count = 0
    face_number = 0

    # Loop through the faces in the video
    while True:
        ret, frame = cap.read()

        if not ret:
            break

    # Convert the frame to grayscale and detect the faces
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through the faces detacted in the frame
    for (x,y,w,h) in faces:
        face_count += 1
        face_number += 1

        face_crop = frame[y:y+h, x:x+w]
        cv2.imwrite(f"face_{face_number}.jpeg", face_crop)
    
    cap.release()
    print(f"{face_count} faces extracted from the video")


vid_file, extract_dir = face_extractor(video_path="../data/video/north.MOV")
print(vid_file, extract_dir)