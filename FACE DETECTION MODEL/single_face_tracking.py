import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('Photos/vid.mp4')
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

face_tracker = None

def fancyDraw(img, bbox, l=10, t=2, rt= 1):
    x, y, w, h = bbox
    x1, y1 = x + w, y + h

    cv2.rectangle(img, bbox, (0, 255, 0), rt)
    # Top Left  x,y
    cv2.line(img, (x, y), (x + l, y), (0, 0, 255), t)
    cv2.line(img, (x, y), (x, y+l), (0, 0, 255), t)
    # Top Right  x1,y
    cv2.line(img, (x1, y), (x1 - l, y), (0, 0, 255), t)
    cv2.line(img, (x1, y), (x1, y+l), (0, 0, 255), t)
    # Bottom Left  x,y1
    cv2.line(img, (x, y1), (x + l, y1), (0, 0, 255), t)
    cv2.line(img, (x, y1), (x, y1 - l), (0, 0, 255), t)
    # Bottom Right  x1,y1
    cv2.line(img, (x1, y1), (x1 - l, y1), (0, 0, 255), t)
    cv2.line(img, (x1, y1), (x1, y1 - l), (0, 0, 255), t)
    return img

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = faceDetection.process(imgRGB)
    if results.detections:
        detection = results.detections[0]
        ih, iw, ic = img.shape
        bboxC = detection.location_data.relative_bounding_box
        bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * 1.6 * ih), int(bboxC.height * ih)

        if face_tracker is None:
            face_tracker = bbox
        else:
            distance = ((face_tracker[0] - bbox[0]) ** 2 + (face_tracker[1] - bbox[1]) ** 2) ** 0.5
            if distance < 100:
                face_tracker = bbox

        # Draw bounding rectangle around the detected face
        img=fancyDraw(img,face_tracker)
        cv2.putText(img, f'Recognition:{int(detection.score[0] * 100)}%',
                    (face_tracker[0], face_tracker[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                    1, (255, 0, 100), 1)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 2)
    cv2.imshow("Window", img)
    cv2.waitKey(1)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break