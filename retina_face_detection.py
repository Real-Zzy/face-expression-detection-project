import cv2
import os
from retinaface import RetinaFace

# Change directory if needed
os.chdir(os.path.dirname(__file__))
print("Current working directory:", os.getcwd())

# Load video
webcam_video_stream = cv2.VideoCapture('videos/test_video.mp4')

if not webcam_video_stream.isOpened():
    print("Error: Could not open video file.")
    exit()

while True:
    ret, current_frame = webcam_video_stream.read()
    if not ret:
        break

    # Resize smaller for speed (optional)
    current_frame_small = cv2.resize(current_frame, (0, 0), fx=0.8, fy=0.8)

    # Detect faces using RetinaFace
    faces = RetinaFace.detect_faces(current_frame_small)

    if isinstance(faces, dict):
        for index, face in enumerate(faces.values()):
            x1, y1, x2, y2 = face['facial_area']  # (x1, y1, x2, y2)

            top_pos = int(y1)
            right_pos = int(x2)
            bottom_pos = int(y2)
            left_pos = int(x1)

            print(f"Found face {index+1} at location Top: {top_pos}, Left: {left_pos}, Bottom: {bottom_pos}, Right: {right_pos}")

            # Blur face
            current_face_image = current_frame_small[top_pos:bottom_pos, left_pos:right_pos]
            if current_face_image.size != 0:
                current_face_image = cv2.GaussianBlur(current_face_image, (99, 99), 30)
                current_frame_small[top_pos:bottom_pos, left_pos:right_pos] = current_face_image

            # Draw rectangle
            cv2.rectangle(current_frame_small, (left_pos, top_pos), (right_pos, bottom_pos), (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Webcam Video - RetinaFace", current_frame_small)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

webcam_video_stream.release()
cv2.destroyAllWindows()
