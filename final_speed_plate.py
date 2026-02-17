import cv2
import time
import math
import pandas as pd
from ultralytics import YOLO

# -------------------- CONFIG --------------------
VIDEO_PATH = "short-test4.mp4"
VEHICLE_MODEL_PATH = "yolov8n.pt"
PLATE_MODEL_PATH = "best.pt"

SPEED_LIMIT = 40  # km/h
DISTANCE_METERS = 10  # Real-world distance between two lines

LINE_Y1 = 300
LINE_Y2 = 450

# -------------------- LOAD MODELS --------------------
vehicle_model = YOLO(VEHICLE_MODEL_PATH)
plate_model = YOLO(PLATE_MODEL_PATH)

# -------------------- VIDEO --------------------
cap = cv2.VideoCapture(VIDEO_PATH)

vehicle_times = {}
overspeed_data = []

def calculate_speed(t1, t2):
    time_diff = t2 - t1
    if time_diff <= 0:
        return 0
    speed_mps = DISTANCE_METERS / time_diff
    return speed_mps * 3.6  # Convert to km/h

# -------------------- MAIN LOOP --------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = vehicle_model(frame)

    for result in results:
        boxes = result.boxes

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # Detect only cars, buses, trucks (COCO IDs)
            if cls in [2, 3, 5, 7] and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                vehicle_id = f"{x1}_{y1}"

                # First Line
                if LINE_Y1 - 5 < cy < LINE_Y1 + 5:
                    vehicle_times[vehicle_id] = time.time()

                # Second Line
                if LINE_Y2 - 5 < cy < LINE_Y2 + 5:
                    if vehicle_id in vehicle_times:
                        t1 = vehicle_times[vehicle_id]
                        t2 = time.time()
                        speed = calculate_speed(t1, t2)

                        color = (0, 255, 0)

                        if speed > SPEED_LIMIT:
                            color = (0, 0, 255)

                            # -------------------- PLATE DETECTION --------------------
                            vehicle_crop = frame[y1:y2, x1:x2]
                            plate_results = plate_model(vehicle_crop)

                            for plate in plate_results:
                                for pbox in plate.boxes:
                                    px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                                    plate_crop = vehicle_crop[py1:py2, px1:px2]

                                    plate_filename = f"plate_{int(time.time())}.jpg"
                                    cv2.imwrite(plate_filename, plate_crop)

                                    overspeed_data.append({
                                        "Speed (km/h)": round(speed, 2),
                                        "Plate Image": plate_filename
                                    })

                        cv2.putText(frame, f"{int(speed)} km/h",
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, color, 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # Draw Speed Lines
    cv2.line(frame, (0, LINE_Y1), (frame.shape[1], LINE_Y1), (255, 0, 0), 2)
    cv2.line(frame, (0, LINE_Y2), (frame.shape[1], LINE_Y2), (255, 0, 0), 2)

    cv2.imshow("Speed + Plate Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------- SAVE EXCEL --------------------
if overspeed_data:
    df = pd.DataFrame(overspeed_data)
    df.to_excel("overspeed_vehicles.xlsx", index=False)

cap.release()
cv2.destroyAllWindows()
