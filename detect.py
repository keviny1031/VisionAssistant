import cv2
from ultralytics import YOLO
from collections import Counter

model = YOLO("yolov8n.pt")  # Load YOLOv8 model
cap = cv2.VideoCapture(0)   # Access webcam

def get_detections():
    ret, frame = cap.read()
    if not ret:
        return {}

    results = model(frame)[0]
    class_ids = results.boxes.cls.tolist()
    class_names = [model.names[int(i)] for i in class_ids]
    counts = dict(Counter(class_names))
    return counts

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO and draw boxes
        results = model(frame)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield frame as multipart stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
