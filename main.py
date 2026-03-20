"""
face_camera_runner.py — Main loop
===================================
ไฟล์นี้เป็นตัวประสานงาน:
  1. อ่านเฟรมจากกล้อง          (camera.py)
  2. ตรวจจับ+จดจำใบหน้า         (face_recognition)
  3. ตรวจสอบ liveness 5 ด่าน     (liveness_engine.py)
  4. จัดการ check-in / check-out  (session_manager.py)
  5. วาด UI                     (ui_renderer.py)

โครงสร้างไฟล์:
  config.py          — ตั้งค่าทั้งหมด (แก้ที่นี่ที่เดียว)
  camera.py          — ThreadedCamera
  liveness_engine.py — Anti-spoofing 5 ด่าน
  session_manager.py — Person tracking + check-in/out
  ui_renderer.py     — วาด UI ทั้งหมด
  attendance_db.py   — เชื่อมต่อฐานข้อมูล (ไม่ได้แก้)
"""

import os
import cv2
import pickle
import numpy as np
import face_recognition
import mediapipe as mp
from datetime import datetime

import config as cfg
from camera import ThreadedCamera
from session_manager import SessionManager
import ui_renderer as ui


def run_camera(camera_index: int = 2, camera_name: str = "CAM_MAIN"):
    """Main loop — เปิดกล้อง, ตรวจหน้า, ตรวจ liveness, บันทึกเวลา"""

    # ─── โหลด face encodings ───
    if not os.path.exists(cfg.ENCODINGS_FILE):
        raise FileNotFoundError(f"ไม่พบ {cfg.ENCODINGS_FILE} — รัน encode_faces.py ก่อน")

    with open(cfg.ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    known_encodings = data["encodings"]
    known_names     = data["names"]

    # ─── เปิดกล้อง ───
    cam = ThreadedCamera(camera_index)

    # ─── MediaPipe Hands (สำหรับ finger challenge) ───
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    # ─── Session Manager ───
    session = SessionManager()

    # ─── แสดงข้อมูลเริ่มต้น ───
    print("=== ระบบตรวจใบหน้า (Multi-Layer Anti-Spoof) ===")
    print(f"[CAM]       {cam.width}x{cam.height}  detect_every={cfg.DETECT_EVERY_N_FRAMES}")
    print(f"[DEPTH]     {cfg.DEPTH_FRAMES_REQUIRED}/{cfg.DEPTH_FRAMES_WINDOW} เฟรม")
    print(f"[TEXTURE]   {'ON' if cfg.TEXTURE_ENABLED else 'OFF'}")
    print(f"[SCREEN]    {'ON' if cfg.SCREEN_DETECT_ENABLED else 'OFF'}")
    print(f"[CHALLENGE] x{cfg.CHALLENGE_COUNT}  timeout={cfg.CHALLENGE_TIMEOUT}s" if cfg.CHALLENGE_ENABLED else "[CHALLENGE] OFF")

    # ─── หน้าต่าง ───
    win_name = "Face Attendance System"
    if cfg.FULLSCREEN:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # ─── ตัวแปร loop ───
    start_ts        = datetime.now().timestamp()
    checkout_done   = False
    frame_count     = 0
    last_results    = []          # cache ผลลัพธ์ face detection
    fps_counter     = 0
    fps_timer       = start_ts
    display_fps     = 0.0

    # ═════════════════════════════════════════
    # MAIN LOOP
    # ═════════════════════════════════════════
    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            continue

        now    = datetime.now()
        now_ts = now.timestamp()
        frame_count += 1

        # ─── FPS ───
        fps_counter += 1
        if now_ts - fps_timer >= 1.0:
            display_fps = fps_counter / (now_ts - fps_timer)
            fps_counter, fps_timer = 0, now_ts

        # ─── Checkout trigger ───
        if cfg.TEST_MODE:
            should_checkout = (now_ts - start_ts) >= cfg.TEST_DURATION_SECONDS
        else:
            should_checkout = now.time() >= cfg.CHECKOUT_TIME

        if should_checkout and not checkout_done:
            checkout_done = True
            session.do_checkout(camera_name, now)

        if cfg.TEST_MODE and checkout_done:
            break

        # ─── Face detection (ทุก N เฟรม) ───
        do_detect = (frame_count % cfg.DETECT_EVERY_N_FRAMES == 0)

        if do_detect:
            small = cv2.resize(frame, (0, 0), fx=cfg.FRAME_SCALE, fy=cfg.FRAME_SCALE)
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locs  = face_recognition.face_locations(rgb)
            encs  = face_recognition.face_encodings(rgb, locs)
            lms   = face_recognition.face_landmarks(rgb, locs)
            last_results = list(zip(encs, locs, lms))

        # ─── Hand detection (เฉพาะเมื่อมี active challenge) ───
        hand_results = None
        has_challenge = any(
            lv.challenge_phase == "active"
            for lv in session.liveness.values()
            if not lv.confirmed and not lv.failed
        )
        if has_challenge and do_detect:
            hand_results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # ─── ประมวลผลแต่ละหน้า ───
        for enc, loc, lm in last_results:
            # จดจำหน้า
            name = _identify_face(enc, known_encodings, known_names)

            # แปลง coordinates กลับเป็น full-size
            t, r, b, l = loc
            top    = int(t / cfg.FRAME_SCALE)
            right  = int(r / cfg.FRAME_SCALE)
            bottom = int(b / cfg.FRAME_SCALE)
            left   = int(l / cfg.FRAME_SCALE)
            face_w = right - left
            face_box = (top, right, bottom, left)

            # ตัดรูปหน้า
            fh, fw = frame.shape[:2]
            pad = 15
            crop = frame[max(0, top-pad):min(fh, bottom+pad),
                         max(0, left-pad):min(fw, right+pad)]

            # Unknown → วาดกรอบแดงแล้วข้าม
            if name == "Unknown":
                ui.draw_face_box(frame, left, top, right, bottom, cfg.Color.UNKNOWN, "Unknown")
                continue

            # ─── จัดการ session + liveness ───
            person, liveness = session.get_or_create(name, now, crop)

            # อัปเดต liveness (5 ด่าน)
            session.engine.update(
                liveness, lm, crop, face_box, face_w,
                frame, hand_results, now_ts, do_detect,
            )

            # ลอง check-in
            session.try_checkin(name, camera_name)

            # ─── วาด UI ───
            color, label = ui.get_face_visual(name, person, liveness)
            ui.draw_face_box(frame, left, top, right, bottom, color, label)
            ui.draw_landmarks(frame, lm, cfg.FRAME_SCALE)
            ui.draw_challenge_overlay(frame, liveness, now_ts)

        # ─── วาด hands + HUD + panel ───
        ui.draw_hands(frame, hand_results)

        remaining = max(0, cfg.TEST_DURATION_SECONDS - int(now_ts - start_ts)) if cfg.TEST_MODE else 0
        ui.draw_hud(frame, display_fps, cfg.TEST_MODE, checkout_done, remaining)

        panel = ui.build_panel(session.persons, session.liveness, frame.shape[0])
        cv2.imshow(win_name, np.hstack([frame, panel]))

        # ─── ออกด้วย q หรือ ESC ───
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    # ─── Cleanup ───
    hands.close()
    cam.release()
    cv2.destroyAllWindows()


def _identify_face(encoding, known_encodings, known_names) -> str:
    """เทียบหน้ากับ encoding ที่รู้จัก — return ชื่อหรือ 'Unknown'"""
    if len(known_encodings) == 0:
        return "Unknown"

    matches   = face_recognition.compare_faces(known_encodings, encoding, tolerance=cfg.FACE_TOLERANCE)
    distances = face_recognition.face_distance(known_encodings, encoding)

    best = distances.argmin()
    if matches[best]:
        return known_names[best]
    return "Unknown"


if __name__ == "__main__":
    run_camera(camera_index=0, camera_name="CAM_MAIN")