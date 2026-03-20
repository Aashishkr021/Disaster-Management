import cv2
import torch

# Load YOLOv5 model (pretrained)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

# Open webcam (0) or replace with video path
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)
    person_count = 0  #reset count for each frame

    # Extract detections
    for *box, conf, cls in results.xyxy[0]:
        if int(cls) == 0:  # Class 0 = person
            person_count += 1
            x1, y1, x2, y2 = map(int, box)
            label = f'Person {conf:.2f}'

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Human Detection - YOLOv5", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
