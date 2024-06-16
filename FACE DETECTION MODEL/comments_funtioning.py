import cv2
import mediapipe as mp
import time

# Initialize the camera capture
cap = cv2.VideoCapture(0)

# Initialize the previous time for calculating FPS
pTime = 0

# Initialize the MediaPipe face detection and drawing utilities
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

# Initialize a dictionary to store face trackers
face_trackers = {}

# Game like boundary box
def fancyDraw(img, bbox, l=20, t=2, rt= 1):
  
    x, y, w, h = bbox
    x1, y1 = x + w, y + h

  
    cv2.rectangle(img, bbox, (0, 255, 0), rt)

  
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
    # Read frame 
    success, img = cap.read()

    # Convert the image to RGB for face detection
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform face detection
    results = faceDetection.process(imgRGB)

    # If faces are detected
    if results.detections:
        for id, detection in enumerate(results.detections):
            # Get the image shape
            ih, iw, ic = img.shape

            # Get the bounding box coordinates
            bboxC = detection.location_data.relative_bounding_box
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * 1.6 * ih), int(bboxC.height * ih)

            # Check if the face is already being tracked
            matched_fid = None
            min_distance = float('inf')
            for fid, (x, y, w, h) in face_trackers.items():
                distance = ((x - bbox[0]) ** 2 + (y - bbox[1]) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    matched_fid = fid

            # If no matched fid, create a new tracker
            if matched_fid is None:
                print("Creating new tracker", len(face_trackers))
                face_trackers[len(face_trackers)] = bbox
            else:
                # Update the tracker
                face_trackers[matched_fid] = bbox

            # Draw a fancy bounding box around the detected face
            img = fancyDraw(img, bbox)

            # Draw the face recognition score
            cv2.putText(img, f'Recognition:{int(detection.score[0] * 100)}%',
                        (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                        1, (255, 0, 100), 1)

            # Draw the face number
            for fid, (x, y, w, h) in face_trackers.items():
                if (x, y, w, h) == bbox:
                    cv2.putText(img, f'Face {id+1}', (x, y - 40), cv2.FONT_HERSHEY_PLAIN,
                                1, (100, 5, 5), 2)

    # Calculate the FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Draw the FPS
    cv2.putText(img, f'FPS:{int(fps)}', (0, 300), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.imshow("Window", img)
    cv2.waitKey(1)

    #for exit

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
