from ultralytics import YOLO
# from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2
import mediapipe as mp


# ip cam serevr
ip = "https://192.168.132.247:8080/video"

# Configure video capture
video = cv2.VideoCapture()
video.open(ip)

# Configure mediapipe hands solution
hand = mp.solutions.hands
Hand = hand.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils


# Visualize vídeo
while True:
    check, img = video.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = Hand.process(imgRGB)
    handsPoints = results.multi_hand_landmarks
    
    # Extract image dimensions
    hight, width, _ = img.shape 
    
    
    hand_points_coord = []
    
    
    if handsPoints:
        for points in handsPoints:
            # Show hand points coordinates in real time
            print(points) 
            # Draw hands lines
            mpDraw.draw_landmarks(img, points, hand.HAND_CONNECTIONS)
            
            for id, coord in enumerate(points.landmark):
                coord_x, coord_y =  int(coord.x* width), int(coord.y*hight)
                cv2.putText(img, str(id), (coord_x, coord_y+10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,100), 3)
                hand_points_coord.append((coord_x,coord_y))
    
    finger_points = [8, 12, 16, 20]
    finger_count = 0
    
    if handsPoints:
        # Logic to count the thumb
        if (hand_points_coord[4][0] > hand_points_coord[2][0]):
            finger_count += 1
        # Logic to count four, fingers less thumb
        for x in finger_points:
            if hand_points_coord[x][1] < hand_points_coord[x-2][1]:
                finger_count += 1
    
    cv2.putText(img, str(finger_count), (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 10, (255,0,100), 15)     
    
    cv2.imshow("img", img)
    cv2.waitKey(1)
    

