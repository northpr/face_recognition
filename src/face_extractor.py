import cv2

def face_extractor(video_path: str, *args):
    video_file = video_path

    if args["extract_path"] is None:
        extract_dir = video_path+ "./data"
    else:
        extract_dir = args["extract_path"]

    cap = cv2.VideoCapture(video_path)

    return (video_file, extract_dir)

vid_file, extract_dir = face_extractor(video_path="Dir test")
print(vid_file, extract_dir)