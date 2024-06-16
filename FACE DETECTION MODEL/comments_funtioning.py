import cv2
import mediapipe as mp
import time

# Initialize the camera capture
cap = cv2.VideoCapture('Photos/vid.mp4')

# Initialize the previous time for FPS calculation
pTime = 0

# Initialize the MediaPipe face detection and drawing utilities
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

# Initialize a dictionary to store face trackers
face_trackers = {}

while True:
    # Read a frame from the camera
    success, img = cap.read()
    
    # Convert the frame to RGB for MediaPipe processing
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the frame using MediaPipe face detection
    results = faceDetection.process(imgRGB)
    
    # If faces are detected
    if results.detections:
        # Iterate over the detected faces
        for id, detection in enumerate(results.detections):
            # Get the image shape
            ih, iw, ic = img.shape
            
            # Get the bounding box of the detected face
            bboxC = detection.location_data.relative_bounding_box
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * 1.6 * ih), int(bboxC.height * ih)

            # Check if the face is already being tracked
            matched_fid = None
            min_distance = float('inf')
            for fid, (x, y, w, h) in face_trackers.items():
                # Calculate the distance between the current face and the tracked face
                distance = ((x - bbox[0]) ** 2 + (y - bbox[1]) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    matched_fid = fid

            # If no matched fid, create a new tracker
            if matched_fid is None:
                print("Creating new tracker", len(face_trackers))
                face_trackers[len(face_trackers)] = bbox
            else:
                # Updating the tracker
                face_trackers[matched_fid] = bbox

            # Drawing a bounding rectangle around the detected face
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
            cv2.putText(img, f'Recognition:{int(detection.score[0] * 100)}%',
                        (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                        2, (0, 0, 0), 2)

            # Dsiplaying face number
            for fid, (x, y, w, h) in face_trackers.items():
                if (x, y, w, h) == bbox:
                    cv2.putText(img, f'Face {id+1}', (x, y - 40), cv2.FONT_HERSHEY_PLAIN,
                                2, (0, 0, 0), 2)

    # Calculating the FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 2)

    # Display
    cv2.imshow("Window", img)
    cv2.waitKey(1)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break