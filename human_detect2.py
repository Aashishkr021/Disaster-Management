import cv2
import torch

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# Load YOLOv5 model (small & fast)
model = torch.hub.load(
    'ultralytics/yolov5',
    'yolov5s',
    pretrained=True
)

model.to(device)
model.eval()

# Use half precision for Jetson Nano (CUDA only)
if device == 'cuda':
    model.half()

# Open camera (USB cam = 0, CSI cam = use GStreamer)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for better FPS on Nano
    frame = cv2.resize(frame, (640, 480))

    # Convert frame to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Inference
    results = model(img, size=640)

    # Parse detections
    for *box, conf, cls in results.xyxy[0]:
        if int(cls) == 0:  # person class
            x1, y1, x2, y2 = map(int, box)

            label = f'Person {conf:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("Human Detection - Jetson Nano", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
