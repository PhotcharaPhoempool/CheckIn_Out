import os
import pickle
import face_recognition

KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "encodings.pkl"

known_encodings = []
known_names = []

if not os.path.exists(KNOWN_FACES_DIR):
    raise FileNotFoundError(f"ไม่พบโฟลเดอร์ {KNOWN_FACES_DIR}")

for person_name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, person_name)

    if not os.path.isdir(person_dir):
        continue

    for file_name in os.listdir(person_dir):
        file_path = os.path.join(person_dir, file_name)

        try:
            image = face_recognition.load_image_file(file_path)
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            if len(face_encodings) == 0:
                print(f"[ข้าม] ไม่พบใบหน้าในไฟล์: {file_path}")
                continue

            if len(face_encodings) > 1:
                print(f"[ข้าม] พบหลายใบหน้าในไฟล์: {file_path}")
                continue

            known_encodings.append(face_encodings[0])
            known_names.append(person_name)
            print(f"[OK] เพิ่มใบหน้า: {person_name} จาก {file_name}")

        except Exception as e:
            print(f"[ERROR] {file_path} -> {e}")

data = {
    "encodings": known_encodings,
    "names": known_names
}

with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump(data, f)

print(f"\nบันทึกข้อมูลใบหน้าเรียบร้อย -> {ENCODINGS_FILE}")
print(f"จำนวนข้อมูลทั้งหมด: {len(known_names)} ใบหน้า")