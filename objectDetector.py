from ultralytics import YOLO
import cv2

# load a pretrained model
model = YOLO("yolov8s.pt") 

photoDetection = False

if(photoDetection):
    # ip cam server
    ip = "https://192.168.132.247:8080/video"

    # Configure video capture
    video = cv2.VideoCapture()
    video.open(ip)

    check, img = video.read()


    results = model.predict(source=img, save=True, show=True)

    print(results)
else:
    results = model.predict(source="0", save=True, show=True)




