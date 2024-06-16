import cv2
import mediapipe as mp
import time

# Initialize the previous time for FPS calculation
pTime = 0

# Import MediaPipe face detection and drawing utilities
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils

# Create a face detection object
faceDetection = mpFaceDetection.FaceDetection()

# Capture video from the web cam
cap = cv2.VideoCapture(0)

# Initialize the face tracker to None
face_tracker = None

# Define a function to draw a shooter game like bounding box around the face
def fancyDraw(img, bbox, l=10, t=2, rt= 1):
    x, y, w, h = bbox
    x1, y1 = x + w, y + h

    # Draw a green rectangle around the face
    cv2.rectangle(img, bbox, (0, 255, 0), rt)
    # Draw lines at the top-left, top-right, bottom-left, and bottom-right corners of the rectangle
    cv2.line(img, (x, y), (x + l, y), (0, 0, 255), t)
    cv2.line(img, (x, y), (x, y+l), (0, 0, 255), t)
    cv2.line(img, (x1, y), (x1 - l, y), (0, 0, 255), t)
    cv2.line(img, (x1, y), (x1, y+l), (0, 0, 255), t)
    cv2.line(img, (x, y1), (x + l, y1), (0, 0, 255), t)
    cv2.line(img, (x, y1), (x, y1 - l), (0, 0, 255), t)
    cv2.line(img, (x1, y1), (x1 - l, y1), (0, 0, 255), t)
    cv2.line(img, (x1, y1), (x1, y1 - l), (0, 0, 255), t)
    return img

while True:
    # Read a frame from the video capture
    success, img = cap.read()
    # Convert the frame to RGB format
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the frame using the face detection object
    results = faceDetection.process(imgRGB)
    if results.detections:
        # Get the first detection (assuming only one face is detected)
        detection = results.detections[0]
        # Get the shape of the image
        ih, iw, ic = img.shape
        # Get the relative bounding box of the face
        bboxC = detection.location_data.relative_bounding_box
        # Calculate the absolute bounding box coordinates
        bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * 1.6 * ih), int(bboxC.height * ih)

        # If the face tracker is None, set it to the current bounding box
        if face_tracker is None:
            face_tracker = bbox
        else:
            # Calculate the distance between the current bounding box and the previous face tracker
            distance = ((face_tracker[0] - bbox[0]) ** 2 + (face_tracker[1] - bbox[1]) ** 2) ** 0.5
            # If the distance is less than 100, update the face tracker
            if distance < 100:
                face_tracker = bbox

        # Draw a fancy bounding box around the detected face
        img=fancyDraw(img,face_tracker)
        # Display the recognition score
        cv2.putText(img, f'Recognition:{int(detection.score[0] * 100)}%',
                    (face_tracker[0], face_tracker[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                    1, (255, 0, 100), 1)

    # Calculate the current time
    cTime = time.time()
    # Calculate the FPS
    fps = 1 / (cTime - pTime)
    pTime = cTime
    # Display the FPS
    cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 2)
    # Display the output image
    cv2.imshow("Window", img)
    cv2.waitKey(1)

    #Code for termination by pressing 'q' key
    if cv2.waitKey(5) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
