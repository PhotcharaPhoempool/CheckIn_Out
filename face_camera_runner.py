import os
import math
import cv2
import pickle
import numpy as np
import face_recognition
from datetime import datetime, time as dtime
from attendance_db import get_employee_by_name, mark_attendance

ENCODINGS_FILE = "encodings.pkl"
TOLERANCE = 0.45
FRAME_SCALE = 0.25
PANEL_WIDTH = 300

# ===== โหมดทดสอบ =====
TEST_MODE = False
TEST_DURATION_SECONDS = 60
# =====================
CHECKOUT_TIME = dtime(22, 0)  # Production: 4 ทุ่ม

# ===== Anti-Spoofing =====
BLINKS_REQUIRED    = 2
EAR_THRESHOLD      = 0.22
EAR_CONSEC_FRAMES  = 2
LIVENESS_TIMEOUT   = 12
LIVENESS_RETRY_AFTER = 8
# =========================


# ===== ระบบสี (Hex) =====
def hex_to_bgr(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))

CLR_UNKNOWN       = hex_to_bgr("#FF3333")
CLR_LIVENESS      = hex_to_bgr("#FFD700")
CLR_LIVENESS_FAIL = hex_to_bgr("#FF6600")
CLR_CHECKED_IN    = hex_to_bgr("#00DC00")
CLR_CHECKED_OUT   = hex_to_bgr("#FF8C00")
CLR_NOT_IN_DB     = hex_to_bgr("#00C8C8")
CLR_PANEL_BG      = hex_to_bgr("#1C1C1C")
CLR_PANEL_HEADER  = hex_to_bgr("#323232")
CLR_DIVIDER       = hex_to_bgr("#3A3A3A")
CLR_TEXT_WHITE    = hex_to_bgr("#FFFFFF")
CLR_TEXT_DIM      = hex_to_bgr("#BEBEBE")
CLR_TEXT_CYAN     = hex_to_bgr("#00E6E6")
CLR_TEXT_MORE     = hex_to_bgr("#969696")
CLR_HUD_TEST      = hex_to_bgr("#00E6E6")
CLR_HUD_DONE      = hex_to_bgr("#FF8C00")
CLR_FACE_PH       = hex_to_bgr("#3C3C3C")
# ========================


def _ear(eye_pts: list) -> float:
    def d(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])
    v1 = d(eye_pts[1], eye_pts[5])
    v2 = d(eye_pts[2], eye_pts[4])
    h  = d(eye_pts[0], eye_pts[3])
    return (v1 + v2) / (2.0 * h) if h > 0 else 0.0


def _scale_pts(pts, scale):
    return [(int(x / scale), int(y / scale)) for x, y in pts]


def _draw_text(img, text, pos, font_scale=0.45, color=None, thickness=1):
    if color is None:
        color = CLR_TEXT_WHITE
    try:
        from PIL import ImageFont, ImageDraw, Image
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        font_paths = [
            "/usr/share/fonts/truetype/thai/Garuda.ttf",
            "/usr/share/fonts/truetype/tlwg/Garuda.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansThai-Regular.otf",
        ]
        font = None
        for fp in font_paths:
            if os.path.exists(fp):
                font = ImageFont.truetype(fp, max(10, int(font_scale * 28)))
                break
        if font is None:
            font = ImageFont.load_default()
        draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
        img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def _build_panel(person_cache: dict, liveness_cache: dict, frame_height: int):
    """
    Panel ขวา — แสดงต่อคน:
      snapshot (รูปแรก), ชื่อ, เวลาเข้า (first_seen), เวลาล่าสุด (last_seen), สถานะ
    """
    panel = np.zeros((frame_height, PANEL_WIDTH, 3), dtype=np.uint8)
    panel[:] = CLR_PANEL_BG

    cv2.rectangle(panel, (0, 0), (PANEL_WIDTH, 38), CLR_PANEL_HEADER, cv2.FILLED)
    cv2.putText(panel, "Detected Persons", (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, CLR_TEXT_WHITE, 1)

    face_w, face_h = 80, 80
    item_h = face_h + 58
    y = 46

    for name, info in list(person_cache.items()):
        if y + item_h > frame_height - 5:
            n_left = len(person_cache) - list(person_cache.keys()).index(name)
            cv2.putText(panel, f"+ {n_left} more...", (8, y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, CLR_TEXT_MORE, 1)
            break

        lv = liveness_cache.get(name, {"confirmed": True, "failed": False, "blink_count": 0})

        # สีกรอบตามสถานะ
        if lv.get("failed"):
            box_color = CLR_LIVENESS_FAIL
        elif not lv.get("confirmed"):
            box_color = CLR_LIVENESS
        elif info.get("checked_out"):
            box_color = CLR_CHECKED_OUT
        elif info.get("checked_in"):
            box_color = CLR_CHECKED_IN
        else:
            box_color = CLR_NOT_IN_DB

        # รูป snapshot (รูปแรกที่ถ่าย)
        x_face = 8
        snap = info.get("snapshot")
        if snap is not None and snap.size > 0:
            try:
                panel[y:y+face_h, x_face:x_face+face_w] = cv2.resize(snap, (face_w, face_h))
            except Exception:
                cv2.rectangle(panel, (x_face, y), (x_face+face_w, y+face_h), CLR_FACE_PH, cv2.FILLED)
        else:
            cv2.rectangle(panel, (x_face, y), (x_face+face_w, y+face_h), CLR_FACE_PH, cv2.FILLED)
        cv2.rectangle(panel, (x_face, y), (x_face+face_w, y+face_h), box_color, 2)

        # ข้อมูลข้างรูป
        tx = x_face + face_w + 8
        _draw_text(panel, name[:14], (tx, y+14), font_scale=0.40, color=CLR_TEXT_CYAN)
        if len(name) > 14:
            _draw_text(panel, name[14:28], (tx, y+29), font_scale=0.40, color=CLR_TEXT_CYAN)

        # เวลา first_seen (IN)
        first = info.get("first_seen")
        if first:
            cv2.putText(panel, f"IN:  {first.strftime('%H:%M:%S')}", (tx, y+face_h-28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, CLR_CHECKED_IN, 1)

        # เวลา last_seen
        last = info.get("last_seen")
        if last:
            cv2.putText(panel, f"LST: {last.strftime('%H:%M:%S')}", (tx, y+face_h-14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, CLR_TEXT_DIM, 1)

        # สถานะ
        if lv.get("failed"):
            sv, sc = "Spoof/Timeout", CLR_LIVENESS_FAIL
        elif not lv.get("confirmed"):
            sv = f"Blink {lv.get('blink_count',0)}/{BLINKS_REQUIRED}"
            sc = CLR_LIVENESS
        elif info.get("checked_out"):
            sv, sc = "IN + OUT", CLR_CHECKED_OUT
        elif info.get("checked_in"):
            sv, sc = "Checked IN", CLR_CHECKED_IN
        else:
            sv, sc = "Not in DB", CLR_NOT_IN_DB
        cv2.putText(panel, sv, (x_face, y+face_h+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, sc, 1)

        y += item_h
        cv2.line(panel, (5, y-4), (PANEL_WIDTH-5, y-4), CLR_DIVIDER, 1)

    return panel


def _get_or_create_person(person_cache, name, now, face_crop):
    """
    เพิ่มคนในครั้งแรก (พร้อมถ่าย snapshot)
    ถ้ามีแล้ว อัพเดตเฉพาะ last_seen — snapshot ไม่เปลี่ยน
    """
    if name not in person_cache:
        person_cache[name] = {
            "employee_id": None,
            "first_seen":  now,
            "last_seen":   now,
            "snapshot":    face_crop.copy() if face_crop is not None and face_crop.size > 0 else None,
            "checked_in":  False,
            "checked_out": False,
        }
    else:
        person_cache[name]["last_seen"] = now


def run_camera(camera_index=0, camera_name="CAM_SINGLE"):
    if not os.path.exists(ENCODINGS_FILE):
        raise FileNotFoundError("ไม่พบ encodings.pkl กรุณารัน encode_faces.py ก่อน")

    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)

    known_encodings = data["encodings"]
    known_names     = data["names"]

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"ไม่สามารถเปิดกล้องได้: index={camera_index}")

    print("=== ระบบตรวจใบหน้า (First-IN / Last-OUT | Anti-Spoof) ===")
    if TEST_MODE:
        print(f"[TEST]  OUT หลัง {TEST_DURATION_SECONDS}s | กด q เพื่อออก")
    else:
        print(f"[PROD]  OUT เวลา {CHECKOUT_TIME.strftime('%H:%M')} | กด q เพื่อออก")
    print(f"[LIVE]  กระพริบตา {BLINKS_REQUIRED} ครั้งเพื่อยืนยัน")

    person_cache   = {}   # {name: {employee_id, first_seen, last_seen, snapshot, checked_in, checked_out}}
    liveness_cache = {}   # {name: {start_ts, blink_count, consec_below, confirmed, failed}}

    start_ts            = datetime.now().timestamp()
    checkout_done       = False
    checkout_show_until = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("อ่านภาพจากกล้องไม่สำเร็จ")
            break

        now    = datetime.now()
        now_ts = now.timestamp()

        # ===== Checkout trigger =====
        if TEST_MODE:
            should_checkout = (now_ts - start_ts) >= TEST_DURATION_SECONDS
        else:
            should_checkout = now.time() >= CHECKOUT_TIME

        if should_checkout and not checkout_done:
            checkout_done       = True
            checkout_show_until = now_ts + 5
            count = 0
            for pname, info in person_cache.items():
                if info.get("checked_in") and not info.get("checked_out"):
                    emp_id    = info.get("employee_id")
                    last_seen = info.get("last_seen", now)
                    if emp_id:
                        # OUT ด้วยเวลา last_seen (ท้ายสุดที่เห็นใบหน้า)
                        ok = mark_attendance(emp_id, "OUT", camera_name, check_time=last_seen)
                        if ok:
                            info["checked_out"] = True
                            count += 1
                            print(f"[OUT] {pname}  last_seen={last_seen.strftime('%H:%M:%S')}")
            print(f"บันทึก OUT {count} คน")

        if TEST_MODE and checkout_done and checkout_show_until and now_ts > checkout_show_until:
            break

        # ===== Face Detection + Landmarks =====
        small     = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        face_locs = face_recognition.face_locations(rgb_small)
        face_encs = face_recognition.face_encodings(rgb_small, face_locs)
        face_lms  = face_recognition.face_landmarks(rgb_small, face_locs)

        for enc, loc, lm in zip(face_encs, face_locs, face_lms):
            matches   = face_recognition.compare_faces(known_encodings, enc, tolerance=TOLERANCE)
            distances = face_recognition.face_distance(known_encodings, enc)

            name = "Unknown"
            if len(distances) > 0:
                best = distances.argmin()
                if matches[best]:
                    name = known_names[best]

            top, right, bottom, left = loc
            top    = int(top    / FRAME_SCALE)
            right  = int(right  / FRAME_SCALE)
            bottom = int(bottom / FRAME_SCALE)
            left   = int(left   / FRAME_SCALE)

            pad = 15
            fh, fw = frame.shape[:2]
            face_crop = frame[max(0, top-pad):min(fh, bottom+pad),
                              max(0, left-pad):min(fw, right+pad)]

            color = CLR_UNKNOWN

            if name == "Unknown":
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom-28), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, "Unknown", (left+5, bottom-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, CLR_TEXT_WHITE, 1)
                continue

            # เพิ่ม/อัพเดต person_cache (snapshot ถ่ายครั้งแรกเท่านั้น)
            _get_or_create_person(person_cache, name, now, face_crop)

            # ===== EAR / Blink =====
            left_eye_pts  = lm.get("left_eye",  [])
            right_eye_pts = lm.get("right_eye", [])
            landmarks_ok  = len(left_eye_pts) == 6 and len(right_eye_pts) == 6

            current_ear = 0.0
            if landmarks_ok:
                current_ear = (_ear(_scale_pts(left_eye_pts,  FRAME_SCALE)) +
                               _ear(_scale_pts(right_eye_pts, FRAME_SCALE))) / 2.0

            # รีเซ็ต liveness หลัง timeout ถ้ายังไม่ check-in
            if (name in liveness_cache
                    and liveness_cache[name]["failed"]
                    and not person_cache[name]["checked_in"]):
                if now_ts - liveness_cache[name]["start_ts"] > LIVENESS_TIMEOUT + LIVENESS_RETRY_AFTER:
                    print(f"[RETRY] {name}")
                    del liveness_cache[name]
                    person_cache[name]["first_seen"] = now

            # สร้าง liveness state ครั้งแรก
            if name not in liveness_cache:
                liveness_cache[name] = {
                    "start_ts":    now_ts,
                    "blink_count": 0,
                    "consec_below": 0,
                    "confirmed":   False,
                    "failed":      False,
                }

            lv = liveness_cache[name]

            # อัพเดต blink state (เฉพาะเมื่อ landmark ตรวจได้)
            if not lv["confirmed"] and not lv["failed"]:
                if now_ts - lv["start_ts"] > LIVENESS_TIMEOUT:
                    lv["failed"] = True
                    print(f"[TIMEOUT] {name}")
                elif landmarks_ok:
                    if current_ear < EAR_THRESHOLD:
                        lv["consec_below"] += 1
                    else:
                        if lv["consec_below"] >= EAR_CONSEC_FRAMES:
                            lv["blink_count"] += 1
                            print(f"[BLINK] {name} #{lv['blink_count']}/{BLINKS_REQUIRED}  EAR={current_ear:.3f}")
                            if lv["blink_count"] >= BLINKS_REQUIRED:
                                lv["confirmed"] = True
                                print(f"[LIVENESS OK] {name}")
                        lv["consec_below"] = 0

            # ===== Liveness ผ่าน + ยังไม่ CHECK-IN → บันทึก IN =====
            if lv["confirmed"] and not person_cache[name]["checked_in"]:
                info       = person_cache[name]
                first_seen = info["first_seen"]
                employee   = get_employee_by_name(name)
                if employee:
                    emp_id = employee[0]
                    # IN ด้วยเวลา first_seen (เวลาแรกที่เห็นใบหน้า)
                    ok = mark_attendance(emp_id, "IN", camera_name, check_time=first_seen)
                    info["employee_id"] = emp_id
                    info["checked_in"]  = ok
                    if ok:
                        print(f"[IN] {name}  first_seen={first_seen.strftime('%H:%M:%S')}")
                else:
                    print(f"[WARN] ไม่พบ '{name}' ในฐานข้อมูล")

            # สีและ label กรอบ
            if lv.get("failed"):
                color = CLR_LIVENESS_FAIL
                label = f"{name} [SPOOF?]"
            elif not lv["confirmed"]:
                color = CLR_LIVENESS
                label = f"{name} [Blink x{BLINKS_REQUIRED - lv['blink_count']}]"
            elif person_cache[name].get("checked_out"):
                color, label = CLR_CHECKED_OUT, name
            elif person_cache[name].get("checked_in"):
                color, label = CLR_CHECKED_IN, name
            else:
                color, label = CLR_NOT_IN_DB, name

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom-28), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left+5, bottom-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, CLR_TEXT_WHITE, 1)

            # จุดตา (debug liveness)
            for eye_pts in (left_eye_pts, right_eye_pts):
                if len(eye_pts) == 6:
                    for px, py in eye_pts:
                        cv2.circle(frame, (int(px/FRAME_SCALE), int(py/FRAME_SCALE)),
                                   2, CLR_LIVENESS, -1)

        # ===== HUD =====
        if TEST_MODE and not checkout_done:
            rem = max(0, TEST_DURATION_SECONDS - int(now_ts - start_ts))
            cv2.putText(frame, f"[TEST] OUT in {rem}s", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, CLR_HUD_TEST, 2)
        elif checkout_done:
            cv2.putText(frame, "Session Ended - OUT Recorded", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, CLR_HUD_DONE, 2)

        # ===== รวม frame + panel =====
        panel    = _build_panel(person_cache, liveness_cache, frame.shape[0])
        combined = np.hstack([frame, panel])

        cv2.imshow("Face Attendance System", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera(camera_index=0, camera_name="CAM_MAIN")
