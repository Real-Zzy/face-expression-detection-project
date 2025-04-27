import cv2
import os
from retinaface import RetinaFace

# Print current working directory
os.chdir(os.path.dirname(__file__))
print("Current working directory:", os.getcwd())

# Load video file
webcam_video_stream = cv2.VideoCapture('videos/speed_cr7.mp4')

# Check if the video was successfully opened
if not webcam_video_stream.isOpened():
    print("Error: Could not open video file.")
    exit()

# Start reading frames from the video
while True:
    # Read a single frame from the video
    ret, current_frame = webcam_video_stream.read()
    if not ret:
        break

    # Resize smaller for speed
    current_frame_small = cv2.resize(current_frame, (0, 0), fx=0.8, fy=0.8)

    # Detect faces using RetinaFace
    faces = RetinaFace.detect_faces(current_frame_small)

    # Check if any faces are detected
    if isinstance(faces, dict):
        for index, face in enumerate(faces.values()):
            x1, y1, x2, y2 = face['facial_area']  # (x1, y1, x2, y2)

            top_pos = int(y1)
            right_pos = int(x2)
            bottom_pos = int(y2)
            left_pos = int(x1)

            # Print detected face location
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

    # Exit loop when 'q' key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

webcam_video_stream.release()
cv2.destroyAllWindows()
