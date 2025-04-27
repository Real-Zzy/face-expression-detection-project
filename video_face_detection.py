import face_recognition
import cv2
import os

os.chdir(os.path.dirname(__file__))
print("Current working directory: ", os.getcwd())

webcam_video_stream = cv2.VideoCapture('videos/test_video.mp4')

if not webcam_video_stream.isOpened():
    print("Error: Could not open video file.")
    exit()
    
all_face_locations = []

while True:
    ret, current_frame = webcam_video_stream.read()
    current_frame_small = cv2.resize(current_frame, (0, 0), fx=0.8, fy=0.8)
    all_face_locations = face_recognition.face_locations(current_frame_small, model = "hog")

    for index, current_face_location in enumerate(all_face_locations):
        top_pos, right_pos, bottom_pos, left_pos = current_face_location
        print("Found face {} at location Top: {}, Left: {}, Bottom: {}, Right: {}".format(index + 1, top_pos, left_pos, bottom_pos, right_pos))
        #blur
        current_face_image = current_frame_small[top_pos:bottom_pos, left_pos:right_pos]
        current_face_image = cv2.GaussianBlur(current_face_image, (99, 99), 30)
        current_frame_small[top_pos:bottom_pos, left_pos:right_pos] = current_face_image
        #draw rectangle
        cv2.rectangle(current_frame_small, (left_pos, top_pos), (right_pos, bottom_pos), (0, 0, 255), 2)
        cv2.imshow("Webcam Video", current_frame_small)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

webcam_video_stream.release()
cv2.destroyAllWindows()