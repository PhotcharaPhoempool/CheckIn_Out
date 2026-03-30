"""
main.py — Main loop (InsightFace / ArcFace)
=============================================
เปลี่ยนจาก face_recognition (dlib 128d) → InsightFace (ArcFace 512d)

ติดตั้ง:
  pip install insightface onnxruntime-gpu   # (หรือ onnxruntime สำหรับ CPU)

ครั้งแรกที่รัน จะ download model buffalo_l อัตโนมัติ (~300MB)
"""

import os
import sys
import glob


def _ensure_nvidia_libs():
    """
    ตั้ง LD_LIBRARY_PATH ให้ชี้ไปที่ CUDA/cuDNN libs ใน venv ก่อน
    เพื่อให้ onnxruntime-gpu ใช้ GPU ได้โดยไม่ crash
    re-exec ตัวเองถ้าต้องการ
    """
    if os.environ.get("_NVIDIA_LIBS_SET"):
        return
    venv_prefix = os.path.dirname(os.path.dirname(sys.executable))
    nvidia_paths = sorted(glob.glob(
        f"{venv_prefix}/lib/python*/site-packages/nvidia/*/lib"
    ))
    if not nvidia_paths:
        return
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    new_path = ":".join(nvidia_paths) + (":" + existing if existing else "")
    os.environ["LD_LIBRARY_PATH"] = new_path
    os.environ["_NVIDIA_LIBS_SET"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.execv(sys.executable, [sys.executable, "-u"] + sys.argv)


_ensure_nvidia_libs()

import cv2
import pickle
import numpy as np
import mediapipe as mp
from datetime import datetime
from numpy.linalg import norm

import config as cfg
from camera import ThreadedCamera
from session_manager import SessionManager
import ui_renderer as ui


# ─── InsightFace landmark → dict (สำหรับ DepthAnalyzer) ──────
def landmarks_68_to_dict(pts) -> dict:
    """
    แปลง 68-point landmarks (array shape 68x2 หรือ 68x3)
    เป็น dict แบบที่ DepthAnalyzer ใช้ (เหมือน face_recognition)

    dlib 68-point mapping:
      0-16   chin (17 points)
      17-21  left_eyebrow (5)
      22-26  right_eyebrow (5)
      27-30  nose_bridge (4)
      31-35  nose_tip (5)
      36-41  left_eye (6)
      42-47  right_eye (6)
    """
    p = [(float(pts[i][0]), float(pts[i][1])) for i in range(68)]
    return {
        "chin":           p[0:17],
        "left_eyebrow":   p[17:22],
        "right_eyebrow":  p[22:27],
        "nose_bridge":    p[27:31],
        "nose_tip":       p[31:36],
        "left_eye":       p[36:42],
        "right_eye":      p[42:48],
    }


# ─── Face Matching (cosine similarity) ──────
def identify_face(embedding, known_norms: np.ndarray, known_names) -> str:
    """เทียบ 512d embedding กับฐานข้อมูล ด้วย cosine similarity (vectorized)
    known_norms: pre-normalized matrix (N, 512) สร้างครั้งเดียวตอน startup"""
    if known_norms is None or len(known_norms) == 0:
        return "Unknown"
    emb_norm = embedding / (norm(embedding) + 1e-10)
    sims = known_norms @ emb_norm          # (N,512) @ (512,) → (N,)
    best_idx = int(np.argmax(sims))
    if sims[best_idx] >= cfg.FACE_TOLERANCE:
        return known_names[best_idx]
    return "Unknown"


def run_camera(camera_index: int = 1, camera_name: str = "CAM_MAIN"):
    """Main loop"""

    # ─── โหลด face encodings (ArcFace 512d) ───
    if not os.path.exists(cfg.ENCODINGS_FILE):
        raise FileNotFoundError(
            f"ไม่พบ {cfg.ENCODINGS_FILE}\n"
            f"รัน encode_faces_arcface.py ก่อน"
        )
    with open(cfg.ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    known_names = data["names"]
    # Pre-normalize embeddings เป็น matrix ครั้งเดียว → ใช้ vectorized dot product
    _raw = np.array(data["encodings"], dtype=np.float32)
    known_norms = _raw / (np.linalg.norm(_raw, axis=1, keepdims=True) + 1e-10)
    print(f"[DB] โหลด {len(known_names)} คน จาก {cfg.ENCODINGS_FILE}")

    # ─── InsightFace ───
    from insightface.app import FaceAnalysis
    import onnxruntime as ort

    # Auto-detect — ใช้เฉพาะ provider ที่พร้อมจริง (ไม่ error อีก)
    # Auto-detect GPU — กรอง TensorRT ออก (ต้องลง TensorRT แยก)
    available = ort.get_available_providers()
    use_providers = [p for p in available if p != "TensorrtExecutionProvider"]
    print(f"[ORT] Using: {use_providers}")

    # ใช้เฉพาะ models ที่จำเป็น — ข้าม genderage + 2d106det (ไม่ได้ใช้ ลด inference 2/5)
    _needed = ["detection", "landmark_3d_68", "recognition"]
    try:
        app = FaceAnalysis(
            name="buffalo_l",
            providers=use_providers,
            allowed_modules=_needed,
        )
    except Exception as e:
        print(f"[WARN] GPU ใช้ไม่ได้: {e}")
        print("[WARN] Fallback → CPU")
        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            allowed_modules=_needed,
        )
    app.prepare(ctx_id=0, det_size=cfg.DET_SIZE)
    print(f"[ARCFACE] InsightFace ready  det_size={cfg.DET_SIZE}")

    # ─── เปิดกล้อง ───
    cam_src = cfg.CAMERA_URL if cfg.CAMERA_URL else camera_index
    if cfg.CAMERA_URL:
        print(f"[CAM] IP camera: {cfg.CAMERA_URL}")
    cam = ThreadedCamera(cam_src)

    # ─── MediaPipe Hands ───
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    # ─── Session Manager ───
    session = SessionManager()

    # ─── แสดงข้อมูลเริ่มต้น ───
    print("=== ระบบตรวจใบหน้า (ArcFace + Multi-Layer Anti-Spoof) ===")
    print(f"[CAM]       {cam.width}x{cam.height}")
    print(f"[DEPTH]     {cfg.DEPTH_FRAMES_REQUIRED}/{cfg.DEPTH_FRAMES_WINDOW}")
    print(f"[TEXTURE]   {'ON' if cfg.TEXTURE_ENABLED else 'OFF'}")
    print(f"[SCREEN]    {'ON' if cfg.SCREEN_DETECT_ENABLED else 'OFF'}")
    if cfg.CHALLENGE_ENABLED:
        print(f"[CHALLENGE] x{cfg.CHALLENGE_COUNT}  timeout={cfg.CHALLENGE_TIMEOUT}s")
    print(f"[FAS]       {'ON' if cfg.FAS_ENABLED else 'OFF'}")

    # ─── หน้าต่าง ───
    win_name = "Face Attendance System"
    if cfg.FULLSCREEN:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # ─── ตัวแปร loop ───
    start_ts     = datetime.now().timestamp()
    checkout_done = False
    frame_count  = 0
    last_faces   = []     # cache ผลลัพธ์ insightface
    fps_counter  = 0
    fps_timer    = start_ts
    display_fps  = 0.0
    last_face_ts = start_ts

    # ─── Screen Debug EMA (TEST_MODE) ───
    # smoothing กันกระพริบ — ค่าตัวเลขใช้ EMA, ค่า bool ใช้ latest
    _EMA_A = 0.20   # alpha: ต่ำ = smooth กว่า (ตอบสนองช้ากว่า)
    _screen_ema: dict = {}   # EMA ของ float values
    _screen_debug_display: dict = {}   # ส่งเข้า build_panel

    # ─── คำนวณ oval ───
    def _compute_oval(h, w):
        cx = w // 2
        cy = int(h * cfg.GUIDE_OVAL_CY)
        ew = int(h * cfg.GUIDE_OVAL_EW)
        eh = int(h * cfg.GUIDE_OVAL_EH)
        return cx, cy, ew, eh

    # ═══════════════════════════════════════
    # MAIN LOOP
    # ═══════════════════════════════════════
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

        # ─── Checkout ───
        should_checkout = (
            (now_ts - start_ts >= cfg.TEST_DURATION_SECONDS) if cfg.TEST_MODE
            else (now.time() >= cfg.CHECKOUT_TIME)
        )
        if should_checkout and not checkout_done:
            checkout_done = True
            session.do_checkout(camera_name, now)
        if cfg.TEST_MODE and checkout_done:
            break

        # ─── Face detection (InsightFace) ───
        do_detect = (frame_count % cfg.DETECT_EVERY_N_FRAMES == 0)
        if do_detect:
            last_faces = app.get(frame)

        # ─── Hand detection ───
        hand_results = None
        has_challenge = any(
            lv.challenge_phase == "active"
            for lv in session.liveness.values()
            if not lv.confirmed and not lv.failed
        )
        if has_challenge:   # ทุกเฟรม (ไม่ขึ้นกับ do_detect) → จับนิ้วแม่นขึ้น
            hand_results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # ─── Reset expired ───
        session.cleanup_expired(now_ts)

        # ─── No-face reset ───
        if last_faces:
            last_face_ts = now_ts
        elif now_ts - last_face_ts >= cfg.NO_FACE_RESET_SEC:
            for name in list(session.liveness.keys()):
                person = session.persons.get(name)
                if not person or not person.checked_in:
                    del session.liveness[name]
            last_faces = []
            last_face_ts = now_ts

        # copy frame เฉพาะเมื่อมีคนที่ยังไม่ได้ snapshot (ประหยัด copy ส่วนใหญ่)
        _need_snapshot = any(
            not p.checked_in
            for p in session.persons.values()
        ) or not session.persons
        orig_frame = frame.copy() if _need_snapshot else frame

        # ─── Oval params ───
        fh, fw = frame.shape[:2]
        oval_cx, oval_cy, oval_ew, oval_eh = _compute_oval(fh, fw)

        # ─── Process faces ───
        _face_with_names = []

        for face in last_faces:
            # ── bbox ──
            x1, y1, x2, y2 = face.bbox.astype(int)
            left, top, right, bottom = x1, y1, x2, y2
            face_w = right - left
            face_box = (top, right, bottom, left)

            # ── ตรวจว่าอยู่ในวงรี ──
            fcx = (left + right) / 2.0
            fcy = (top + bottom) / 2.0
            in_oval = (
                ((fcx - oval_cx) / oval_ew) ** 2 +
                ((fcy - oval_cy) / oval_eh) ** 2
            ) <= cfg.GUIDE_IN_OVAL_TOL

            if not in_oval:
                ui.draw_face_box(frame, left, top, right, bottom,
                                 cfg.Color.UNKNOWN, "Move into oval")
                continue

            # ── Identify (ArcFace 512d) ──
            embedding = face.embedding
            name = identify_face(embedding, known_norms, known_names)

            _face_with_names.append((face_box, name))

            # ── Face crop ──
            pad = 15
            crop = orig_frame[max(0, top-pad):min(fh, bottom+pad),
                              max(0, left-pad):min(fw, right+pad)]

            if name == "Unknown":
                ui.draw_face_box(frame, left, top, right, bottom,
                                 cfg.Color.UNKNOWN, "Unknown")
                continue

            # ── Landmarks (68-point → dict) ──
            lm_dict = None
            if face.landmark_3d_68 is not None:
                lm_dict = landmarks_68_to_dict(face.landmark_3d_68)
            elif face.landmark_2d_106 is not None:
                # fallback: ใช้ 5-point จาก kps (ไม่ครบ แต่ดีกว่าไม่มี)
                lm_dict = None  # DepthAnalyzer จะข้ามถ้าไม่มี

            # ── Liveness ──
            person, liveness = session.get_or_create(name, now, crop)

            if lm_dict:
                session.engine.update(
                    liveness, lm_dict, crop, face_box, face_w,
                    frame, hand_results, now_ts, do_detect,
                )
            else:
                # ไม่มี 68-point landmarks → ข้าม depth/motion แต่ยัง check texture/screen/FAS
                session.engine.update(
                    liveness, {}, crop, face_box, face_w,
                    frame, hand_results, now_ts, do_detect,
                )

            if liveness.confirmed and not person.checked_in:
                person.snapshot = orig_frame.copy()

            session.try_checkin(name, camera_name)

            # ── Screen Debug (TEST_MODE) — เก็บค่าล่าสุด อัปเดต EMA ทีหลัง ──
            if cfg.TEST_MODE and do_detect:
                _raw_dbg = session.engine.screen.detect_debug(frame, face_box)
                if _raw_dbg.get("valid"):
                    _float_keys = ["ratio", "inner_density",
                                   "fft_score", "fft_peak_rate", "fft_radial_cov"]
                    for _k in _float_keys:
                        _v = _raw_dbg.get(_k, 0.0)
                        _screen_ema[_k] = (_EMA_A * _v
                                           + (1 - _EMA_A) * _screen_ema.get(_k, _v))
                    # ดึง timing จาก liveness state ของคนแรกที่เจอ
                    _lv = next(iter(session.liveness.values()), None)
                    _scr_dur  = (now_ts - _lv.screen_detect_start_ts
                                 if _lv and _lv.screen_detect_start_ts > 0 else 0.0)
                    _real_dur = (now_ts - _lv.screen_real_start_ts
                                 if _lv and _lv.screen_real_start_ts > 0 else 0.0)

                    _screen_debug_display = {
                        "valid":          True,
                        "ratio":          _screen_ema.get("ratio", 0.0),
                        "threshold":      _raw_dbg.get("threshold", cfg.SCREEN_EDGE_MAX),
                        "inner_density":  _screen_ema.get("inner_density", 0.0),
                        "inner_thresh":   _raw_dbg.get("inner_thresh", cfg.SCREEN_INNER_MAX),
                        "fft_score":      _screen_ema.get("fft_score", 1.0),
                        "fft_thresh":     _raw_dbg.get("fft_thresh", cfg.FFT_SCORE_MIN),
                        "fft_peak_rate":  _screen_ema.get("fft_peak_rate", 0.0),
                        "fft_radial_cov": _screen_ema.get("fft_radial_cov", 0.0),
                        "screen_timer":   _scr_dur,
                        "real_timer":     _real_dur,
                        "screen_confirm": cfg.SCREEN_CONFIRM_SEC,
                        "real_reset":     cfg.SCREEN_RESET_SEC,
                        # bool ใช้ค่าล่าสุดตรงๆ (ไม่ smooth)
                        "is_border":  _raw_dbg.get("is_border", False),
                        "is_inner":   _raw_dbg.get("is_inner", False),
                        "is_fft":     _raw_dbg.get("is_fft", False),
                        "is_screen":  _raw_dbg.get("is_screen", False),
                    }

            # ── Draw ──
            if cfg.SHOW_LANDMARKS and lm_dict:
                ui.draw_landmarks(frame, lm_dict, scale=1.0)

            color, label = ui.get_face_visual(name, person, liveness)
            ui.draw_face_box(frame, left, top, right, bottom, color, label)

        # ─── Guide overlay ───
        ui.draw_face_guide(frame, _face_with_names,
                           session.liveness, session.persons, now_ts)

        # ─── Hands + HUD + Panel ───
        ui.draw_hands(frame, hand_results)

        remaining = max(0, cfg.TEST_DURATION_SECONDS - int(now_ts - start_ts)) if cfg.TEST_MODE else 0
        ui.draw_hud(frame, display_fps, cfg.TEST_MODE, checkout_done, remaining)

        panel = ui.build_panel(session.persons, session.liveness, frame.shape[0],
                               screen_debug=_screen_debug_display if cfg.TEST_MODE else None)
        cv2.imshow(win_name, np.hstack([frame, panel]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    hands.close()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera(camera_index=1, camera_name="CAM_MAIN")