import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

# Initialize face trackers
face_trackers = {}

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = faceDetection.process(imgRGB)
    if results.detections:
        for id, detection in enumerate(results.detections):
            ih, iw, ic = img.shape
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

            # Draw bounding rectangle around the detected face
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
            cv2.putText(img, f'Recognition:{int(detection.score[0] * 100)}%',
                        (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                        2, (0, 0, 0), 2)
            cv2.putText(img, f'Face {matched_fid}', (bbox[0], bbox[1] - 40), cv2.FONT_HERSHEY_PLAIN,
                        2, (0, 0, 0), 2)

    # Draw the face trackers
    for fid, (x, y, w, h) in face_trackers.items():
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, f'Face {fid}', (x, y - 40), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 0, 0), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 2)
    cv2.imshow("Window", img)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break