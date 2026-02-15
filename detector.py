#libraries
import cv2

from ultralytics import YOLO


# On first run this downloads yolov8n.pt (~6MB) automatically
model = YOLO("yolov8n.pt")  # use .pt on Mac, NCNN on Pi

# 0 = built-in webcam, try 1 or 2 if you have external cameras
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Camera opened. Press Q to quit.")

frame_count = 0
last_results = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run inference every other frame
    if frame_count % 2 == 0:
        last_results = model(frame, classes=[0], verbose=False, imgsz=640)

    # Draw bounding boxes
    for result in last_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            if conf > 0.5:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {conf:.0%}",
                            (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)

    count = sum(
        1 for r in last_results
        for b in r.boxes
        if float(b.conf[0]) > 0.5
    )
    cv2.putText(frame, f"People: {count}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("People Detector", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

