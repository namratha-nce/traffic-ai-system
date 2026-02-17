import cv2
import math
import os
import easyocr
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from tkinter import Tk, filedialog

# ================= SETTINGS =================
SPEED_LIMIT = 40
PIXELS_PER_METER = 8.8
FPS = 18
IMAGE_DIR = "images"
# ============================================

os.makedirs(IMAGE_DIR, exist_ok=True)

# ---------------- File Upload ----------------
root = Tk()
root.withdraw()
video_path = filedialog.askopenfilename(
    title="Select Traffic Video",
    filetypes=[("Video Files", "*.mp4 *.avi")]
)

if not video_path:
    print("No video selected.")
    exit()
print("Selected Video:", video_path)

# ---------------- Load ML Models ----------------

print("Loading YOLO vehicle model...")
vehicle_model = YOLO("yolov8n.pt")  # auto downloads if not present

print("Loading Plate Detection model...")
plate_model = YOLO("best.pt")  # Your trained plate model

print("Loading DeepSORT tracker...")
tracker = DeepSort(max_age=30)

print("Loading OCR model...")
reader = easyocr.Reader(['en'])

video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("Could not open video!")
    exit()

# ---------------- Speed Function ----------------
def estimate_speed(p1, p2):
    d_pixels = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
    d_meters = d_pixels / PIXELS_PER_METER
    return d_meters * FPS * 3.6

# ---------------- Tracking Dictionaries ----------------
car_positions = {}
car_plate = {}

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = video.read()
    if not ret:
        break

    results = vehicle_model(frame)

    detections = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = vehicle_model.names[cls]

            if label in ["car", "truck", "bus"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                detections.append(([x1, y1, x2-x1, y2-y1], conf, label))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())

        cx = int((l + r) / 2)
        cy = int((t + b) / 2)

        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (l, t-40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        if track_id not in car_positions:
            car_positions[track_id] = (cx, cy)
            car_plate[track_id] = ""
            continue

        speed = estimate_speed(car_positions[track_id], (cx, cy))
        car_positions[track_id] = (cx, cy)

        cv2.putText(frame, f"{int(speed)} km/h", (l, t-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # ---------------- Overspeed Condition ----------------
        if speed > SPEED_LIMIT and car_plate[track_id] == "":

            car_crop = frame[t:b, l:r]
            img_path = os.path.join(IMAGE_DIR, f"{track_id}.jpg")
            cv2.imwrite(img_path, car_crop)

            # Plate Detection
            plate_results = plate_model(car_crop)

            for plate_res in plate_results:
                for plate_box in plate_res.boxes:
                    px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
                    plate_crop = car_crop[py1:py2, px1:px2]

                    # OCR
                    ocr_result = reader.readtext(plate_crop)

                    if ocr_result:
                        plate_number = ocr_result[0][1]
                        car_plate[track_id] = plate_number

        # Show plate
        if car_plate[track_id]:
            cv2.putText(frame, f"Plate: {car_plate[track_id]}",
                        (l, t-65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("AI Overspeed & Plate Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()
