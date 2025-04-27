# import face_recognition
import cv2
# from tensorflow.keras.preprocessing import image
import numpy as np
import os
from torchvision import models, transforms
import torch
import torch.nn as nn
from retinaface import RetinaFace


os.chdir(os.path.dirname(__file__))
print("Current working directory: ", os.getcwd())

webcam_video_stream = cv2.VideoCapture("videos/speed_cr7.mp4")

# face_exp_model = model_from_json(open("models/facial_expression_model_structure.json", "r").read())
# face_exp_model.load_weights('models/facial_expression_model_weights.h5')
epochs=50
model_path = f"models/fer_resnet50_epoch{epochs}.pth"
face_exp_model = models.resnet50(weights="IMAGENET1K_V2")
face_exp_model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # already 3-ch
num_ftrs = face_exp_model.fc.in_features
face_exp_model.fc  = nn.Linear(num_ftrs, 7) 
face_exp_model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
face_exp_model = face_exp_model.to(device)
face_exp_model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),   # 1→3 channels
    transforms.Resize(48),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

emotions_label = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

all_face_locations = []

while True:
    ret, current_frame = webcam_video_stream.read()
    current_frame_small = cv2.resize(current_frame, (0, 0), fx=0.8, fy=0.8)
    # all_face_locations = face_recognition.face_locations(current_frame_small, model = "cnn")
    all_face_locations = RetinaFace.detect_faces(current_frame_small)

    for index, current_face_location in enumerate(all_face_locations.values()):
        x1, y1, x2, y2 = current_face_location['facial_area']
        top_pos = int(y1)
        right_pos = int(x2)
        bottom_pos = int(y2)
        left_pos = int(x1)
        
        print("Found face {} at location Top: {}, Left: {}, Bottom: {}, Right: {}".format(index + 1, top_pos, left_pos, bottom_pos, right_pos))
        current_face_image = current_frame_small[top_pos:bottom_pos, left_pos:right_pos]
        print(current_face_image.shape)
        #draw rectangle
        # cv2.rectangle(current_frame_small, (left_pos, top_pos), (right_pos, bottom_pos), (0, 0, 255), 2)
        #cv2.imshow("Webcam Video", current_frame_small)

        #blur
        # current_face_image = cv2.GaussianBlur(current_face_image, (99, 99), 30)
        # current_frame_small[top_pos:bottom_pos, left_pos:right_pos] = current_face_image

        # current_face_image = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY)
        # current_face_image = cv2.resize(current_face_image, (48, 48))
        # img_pixels = image.img_to_array(current_face_image)
        # img_pixels = np.expand_dims(img_pixels, axis=0)
        # img_pixels /= 255
        
        input_tensor = transform(current_face_image).unsqueeze(0).to(device)
            
        with torch.no_grad():
            outputs = face_exp_model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            emotion_label = emotions_label[predicted.item()]
        
        cv2.rectangle(current_frame, 
                     (left_pos, top_pos),
                     (right_pos, bottom_pos),
                     (0, 0, 255), 2)

        # exp_predictions = face_exp_model.predict(img_pixels)
        # max_index = np.argmax(exp_predictions[0])
        # emotion_label = emotions_label[max_index]

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame_small, emotion_label, (left_pos, bottom_pos), font, 0.5, (255, 255, 255), 1)

    cv2.imshow("Webcam Video", current_frame_small)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

webcam_video_stream.release()
cv2.destroyAllWindows()