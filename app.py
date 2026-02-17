import cv2
import time
import os
import pandas as pd
from ultralytics import YOLO
from flask import Flask, render_template, Response

# ---------------- CONFIG ----------------
VIDEO_PATH = "short-test4.mp4"   # MUST use video file on Render
VEHICLE_MODEL_PATH = "yolov8n.pt"
PLATE_MODEL_PATH = "best.pt"

SPEED_LIMIT = 40
DISTANCE_METERS = 10
LINE_Y1 = 300
LINE_Y2 = 450

app = Flask(__name__)

os.makedirs("static/plates", exist_ok=True)

vehicle_model = YOLO(VEHICLE_MODEL_PATH)
plate_model = YOLO(PLATE_MODEL_PATH)

vehicle_entry_time = {}
processed_ids = set()
overspeed_records = []

def calculate_speed(t1, t2):
    diff = t2 - t1
    if diff <= 0:
        return 0
    return (DISTANCE_METERS / diff) * 3.6

def generate_frames():
    cap = cv2.VideoCapture(VIDEO_PATH)

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = vehicle_model.track(frame, persist=True)

        for result in results:
            boxes = result.boxes

            if boxes.id is None:
                continue

            for box, track_id in zip(boxes, boxes.id):

                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls in [2, 3, 5, 7] and conf > 0.5:

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cy = int((y1 + y2) / 2)
                    track_id = int(track_id)

                    # First line
                    if LINE_Y1 - 5 < cy < LINE_Y1 + 5:
                        vehicle_entry_time[track_id] = time.time()

                    # Second line
                    if LINE_Y2 - 5 < cy < LINE_Y2 + 5:

                        if track_id in vehicle_entry_time and track_id not in processed_ids:

                            speed = calculate_speed(
                                vehicle_entry_time[track_id],
                                time.time()
                            )

                            if speed > SPEED_LIMIT:

                                vehicle_crop = frame[y1:y2, x1:x2]
                                plate_results = plate_model(vehicle_crop)

                                plate_text = "Detected"
                                plate_img_filename = None

                                for plate in plate_results:
                                    for pbox in plate.boxes:

                                        px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                                        plate_crop = vehicle_crop[py1:py2, px1:px2]

                                        plate_img_filename = f"plate_{track_id}.jpg"
                                        plate_img_path = f"static/plates/{plate_img_filename}"

                                        cv2.imwrite(plate_img_path, plate_crop)

                                overspeed_records.append({
                                    "Vehicle ID": track_id,
                                    "Speed": round(speed, 2),
                                    "Plate": plate_text,
                                    "Image": plate_img_filename
                                })

                                processed_ids.add(track_id)

                            cv2.putText(
                                frame,
                                f"{int(speed)} km/h",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 0, 255),
                                2
                            )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 255, 0),
                        2
                    )

        cv2.line(frame, (0, LINE_Y1), (frame.shape[1], LINE_Y1), (255, 0, 0), 2)
        cv2.line(frame, (0, LINE_Y2), (frame.shape[1], LINE_Y2), (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )

@app.route('/')
def index():
    return render_template('index.html', records=overspeed_records)

@app.route('/video')
def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

