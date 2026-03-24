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
    last_face_ts    = start_ts    # เวลาล่าสุดที่เห็นหน้า

    # ═════════════════════════════════════════
    # MAIN LOOP
    # ═════════════════════════════════════════
    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            continue
        if cfg.CAMERA_FLIP:
            frame = cv2.flip(frame, 1)

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

        # ─── Reset liveness state ที่ timeout ครบแล้ว ───
        session.cleanup_expired(now_ts)

        # ─── ไม่เจอหน้า N วินาที → reset liveness (เก็บ snapshot ไว้) ───
        if last_results:
            last_face_ts = now_ts
        elif now_ts - last_face_ts >= cfg.NO_FACE_RESET_SEC:
            # เคลียร์ liveness ของคนที่ยังไม่ checked_in
            for name in list(session.liveness.keys()):
                person = session.persons.get(name)
                if not person or not person.checked_in:
                    del session.liveness[name]
            last_results = []
            last_face_ts = now_ts  # reset timer

        # ─── เก็บ frame ดิบ (สำหรับ snapshot ไม่ติด overlay) ───
        orig_frame = frame.copy()

        # ─── คำนวณ face boxes + ชื่อ (สำหรับ guide) ───
        _face_with_names = []   # [(face_box, name), ...]

        # ─── คำนวณ oval guide (ค่าเดียวกับ ui_renderer.draw_face_guide) ───
        _fh, _fw = frame.shape[:2]
        _oval_cx = _fw // 2
        _oval_cy = int(_fh * 0.47)
        _oval_ew = int(_fh * 0.29)
        _oval_eh = int(_fh * 0.34)

        # ─── ประมวลผลแต่ละหน้า (update liveness ก่อน แล้วค่อย draw guide) ───
        for enc, loc, lm in last_results:
            name = _identify_face(enc, known_encodings, known_names)

            t, r, b, l = loc
            top    = int(t / cfg.FRAME_SCALE)
            right  = int(r / cfg.FRAME_SCALE)
            bottom = int(b / cfg.FRAME_SCALE)
            left   = int(l / cfg.FRAME_SCALE)
            face_w = right - left
            face_box = (top, right, bottom, left)

            # ── ตรวจว่าหน้าอยู่ในวงรีไหม (tolerance 1.4) ──
            fcx = (left + right) / 2.0
            fcy = (top  + bottom) / 2.0
            in_oval = (((fcx - _oval_cx) / _oval_ew) ** 2 +
                       ((fcy - _oval_cy) / _oval_eh) ** 2) <= 1.4

            if not in_oval:
                # เห็นนอกวงรี → วาดกรอบแดงบางๆ แต่ไม่ประมวลผล liveness
                ui.draw_face_box(frame, left, top, right, bottom,
                                 cfg.Color.UNKNOWN, "Move into oval")
                continue

            _face_with_names.append((face_box, name))

            fh, fw = orig_frame.shape[:2]
            pad  = 15
            crop = orig_frame[max(0, top-pad):min(fh, bottom+pad),
                              max(0, left-pad):min(fw, right+pad)]

            if name == "Unknown":
                ui.draw_face_box(frame, left, top, right, bottom, cfg.Color.UNKNOWN, "Unknown")
                continue

            person, liveness = session.get_or_create(name, now, crop)

            session.engine.update(
                liveness, lm, crop, face_box, face_w,
                frame, hand_results, now_ts, do_detect,
            )

            # อัปเดต snapshot ด้วยรูป full frame ตอน liveness ผ่าน (ก่อน check-in)
            if liveness.confirmed and not person.checked_in:
                person.snapshot = orig_frame.copy()

            session.try_checkin(name, camera_name)

            if cfg.SHOW_LANDMARKS:
                ui.draw_landmarks(frame, lm, cfg.FRAME_SCALE)
            color, label = ui.get_face_visual(name, person, liveness)
            ui.draw_face_box(frame, left, top, right, bottom, color, label)

        # ─── Face guide overlay ───
        ui.draw_face_guide(frame, _face_with_names, session.liveness, session.persons, now_ts)

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
    run_camera(camera_index=1, camera_name="CAM_MAIN")