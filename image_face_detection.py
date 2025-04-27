import face_recognition
import cv2

image_to_detect = cv2.imread('./images/MSN.jpg')


if image_to_detect is None:
    print("Image not found!")
else:
    cv2.imshow("test", image_to_detect)

    print("Press 'q' to close the image window.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

all_face_locations = face_recognition.face_locations(image_to_detect, model = "cnn")

face_num = len(all_face_locations)
print ("There are {} face(s) in this image".format(face_num))

#loop through faces
for index, current_face_location in enumerate(all_face_locations):
    top_pos, right_pos, bottom_pos, left_pos = current_face_location
    print("Found face {} at location Top: {}, Left: {}, Bottom: {}, Right: {}".format(index + 1, top_pos, left_pos, bottom_pos, right_pos))
    current_face_image = image_to_detect[top_pos:bottom_pos, left_pos:right_pos]
    if current_face_image is None:
        print("Image not found!")
    else:
        cv2.imshow("Face Number: " + str(index + 1), current_face_image)
        
if (face_num > 0):
    print("Press 'q' to close the image window.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

