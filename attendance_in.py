import os
import csv
import cv2
import pickle
import face_recognition
from datetime import datetime
from db import get_connection
from attendance_db import mark_attendance

ENCODINGS_FILE = "encodings.pkl"
ATTENDANCE_FILE = "attendance.csv"
TOLERANCE = 0.45
FRAME_SCALE = 0.25

if not os.path.exists(ENCODINGS_FILE):
    raise FileNotFoundError("ไม่พบ encodings.pkl กรุณารัน encode_faces.py ก่อน")

with open(ENCODINGS_FILE, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]


def get_employee_by_name(full_name):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, employee_code, full_name
                FROM employees
                WHERE full_name = %s
                LIMIT 1
            """, (full_name,))
            return cur.fetchone()
        
"""def mark_attendance(name):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    file_exists = os.path.exists(ATTENDANCE_FILE)
    already_marked = False

    if file_exists:
        with open(ATTENDANCE_FILE, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 3 and row[0] == name and row[1] == date_str:
                    already_marked = True
                    break

    if not already_marked:
        with open(ATTENDANCE_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["name", "date", "time"])
            writer.writerow([name, date_str, time_str])
        print(f"[ATTENDANCE] {name} เวลา {time_str}") """

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("ไม่สามารถเปิดกล้องได้")

print("เริ่มระบบตรวจใบหน้า กด q เพื่อออก")

recognized_cache = {}

while True:
    ret, frame = cap.read()
    if not ret:
        print("อ่านภาพจากกล้องไม่สำเร็จ")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(
            known_encodings,
            face_encoding,
            tolerance=TOLERANCE
        )
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        name = "Unknown"

        if len(face_distances) > 0:
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = known_names[best_match_index]

        top, right, bottom, left = face_location

        top = int(top / FRAME_SCALE)
        right = int(right / FRAME_SCALE)
        bottom = int(bottom / FRAME_SCALE)
        left = int(left / FRAME_SCALE)

        color = (0, 0, 255)

        if name != "Unknown":
            current_time = datetime.now().timestamp()
            last_seen = recognized_cache.get(name, 0)

        if current_time - last_seen > 10:
            employee = get_employee_by_name(name)
            if employee:
                employee_id = employee[0]
                mark_attendance(employee_id, "IN", "CAM_IN")
                recognized_cache[name] = current_time
            else:
                print(f"ไม่พบพนักงานชื่อ {name} ในฐานข้อมูล")

            color = (0, 255, 0)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
        cv2.putText(
            frame,
            name,
            (left + 6, bottom - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    cv2.imshow("Face Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

from face_camera_runner import run_camera

if __name__ == "__main__":
    run_camera(
        status="IN",
        camera_index=0,
        camera_name="CAM_IN"
    )