import os
import cv2
import pickle
import face_recognition
from datetime import datetime
from attendance_db import get_employee_by_name, mark_attendance

ENCODINGS_FILE = "encodings.pkl"
TOLERANCE = 0.45
FRAME_SCALE = 0.25

def run_camera(status, camera_index, camera_name):
    if not os.path.exists(ENCODINGS_FILE):
        raise FileNotFoundError("ไม่พบ encodings.pkl กรุณารัน encode_faces.py ก่อน")

    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)

    known_encodings = data["encodings"]
    known_names = data["names"]

    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError(f"ไม่สามารถเปิดกล้องได้: index={camera_index}")

    print(f"เริ่มระบบตรวจใบหน้า [{status}] จากกล้อง {camera_name} กด q เพื่อออก")

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
                        success = mark_attendance(employee_id, status, camera_name)
                        if success:
                            recognized_cache[name] = current_time
                    else:
                        print(f"ไม่พบพนักงานชื่อ {name} ในฐานข้อมูล")

                color = (0, 255, 0)

            label = f"{name} [{status}]"

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
            cv2.putText(
                frame,
                label,
                (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        cv2.imshow(f"Face Attendance - {camera_name}", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()