"""
main.py (Windows Version)
=========================
เหมือน Linux แต่ปรับ GPU setup สำหรับ Windows:
  - ใช้ os.add_dll_directory() แทน LD_LIBRARY_PATH + execv
  - ไม่ต้อง re-exec ตัวเอง (Windows โหลด DLL ได้ทันที)

รันจาก Win_Ver/:
  python main.py                  ← ใช้ค่า default
  python main.py cam_main         ← โหลด profiles/cam_main.py
  python main.py test             ← โหลด profiles/test.py
"""

import os
import sys
import glob

# ── Import shared modules จาก root FaceReg ───────────────────────────────────
# Win_Ver/ import logic files จาก parent โดยไม่ต้อง copy ซ้ำ
_WIN_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_WIN_DIR)
# Win_Ver มาก่อน (ได้ config.py, camera.py ของ Windows)
# Root มาทีหลัง (ได้ liveness_engine.py, ui_renderer.py ฯลฯ)
if _WIN_DIR not in sys.path:
    sys.path.insert(0, _WIN_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.append(_ROOT_DIR)


def _setup_win_gpu():
    """
    Windows GPU setup — เพิ่ม CUDA/cuDNN DLL paths ให้ onnxruntime-gpu
    ใช้ os.add_dll_directory() ซึ่งทำงานได้ทันทีโดยไม่ต้อง re-exec
    """
    if os.environ.get("_NVIDIA_LIBS_SET"):
        return
    os.environ["_NVIDIA_LIBS_SET"] = "1"

    venv = os.path.dirname(os.path.dirname(sys.executable))
    added = []

    # หา nvidia DLLs ใน venv (pip install onnxruntime-gpu จะวางไว้ที่นี่)
    for pattern in [
        os.path.join(venv, "Lib", "site-packages", "nvidia", "*", "bin"),
        os.path.join(venv, "Lib", "site-packages", "nvidia", "*", "lib"),
        os.path.join(venv, "Lib", "site-packages", "onnxruntime", "capi"),
    ]:
        for d in glob.glob(pattern):
            if os.path.isdir(d):
                try:
                    os.add_dll_directory(d)
                    added.append(d)
                except OSError:
                    pass

    # หา CUDA จาก system (CUDA_PATH ถูกตั้งโดย CUDA Toolkit installer)
    cuda_home = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME", "")
    if cuda_home:
        for sub in ["bin", "libnvvp"]:
            p = os.path.join(cuda_home, sub)
            if os.path.isdir(p):
                try:
                    os.add_dll_directory(p)
                    added.append(p)
                except OSError:
                    pass

    if added:
        print(f"[GPU] เพิ่ม DLL paths: {len(added)} รายการ")
    else:
        print("[GPU] ไม่พบ NVIDIA DLLs — จะใช้ CPU")


_setup_win_gpu()

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
    if known_norms is None or len(known_norms) == 0:
        return "Unknown"
    emb_norm = embedding / (norm(embedding) + 1e-10)
    sims = known_norms @ emb_norm
    best_idx = int(np.argmax(sims))
    if sims[best_idx] >= cfg.FACE_TOLERANCE:
        return known_names[best_idx]
    return "Unknown"


def run_camera(camera_index: int = 0, camera_name: str = "CAM_MAIN"):
    """Main loop (Windows)"""

    # ─── โหลด face encodings ───
    if not os.path.exists(cfg.ENCODINGS_FILE):
        raise FileNotFoundError(
            f"ไม่พบ {cfg.ENCODINGS_FILE}\n"
            f"รัน encode_faces_arcface.py ก่อน"
        )
    with open(cfg.ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    known_names = data["names"]
    _raw = np.array(data["encodings"], dtype=np.float32)
    known_norms = _raw / (np.linalg.norm(_raw, axis=1, keepdims=True) + 1e-10)
    print(f"[DB] โหลด {len(known_names)} คน จาก {cfg.ENCODINGS_FILE}")

    # ─── InsightFace ───
    from insightface.app import FaceAnalysis
    import onnxruntime as ort

    available = ort.get_available_providers()
    use_providers = [p for p in available if p != "TensorrtExecutionProvider"]
    print(f"[ORT] Using: {use_providers}")

    _needed = ["detection", "landmark_3d_68", "recognition"]
    try:
        app = FaceAnalysis(
            name="buffalo_l",
            providers=use_providers,
            allowed_modules=_needed,
        )
    except Exception as e:
        print(f"[WARN] GPU ใช้ไม่ได้: {e} → Fallback CPU")
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

    print("=== ระบบตรวจใบหน้า (Windows — ArcFace + Anti-Spoof) ===")
    print(f"[CAM]       {cam.width}x{cam.height}")
    print(f"[CHALLENGE] x{cfg.CHALLENGE_COUNT}  timeout={cfg.CHALLENGE_TIMEOUT}s")
    print(f"[FAS]       {'ON' if cfg.FAS_ENABLED else 'OFF'}")

    win_name = "Face Attendance System"
    if cfg.FULLSCREEN:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    start_ts      = datetime.now().timestamp()
    checkout_done = False
    frame_count   = 0
    last_faces    = []
    fps_counter   = 0
    fps_timer     = start_ts
    display_fps   = 0.0
    last_face_ts  = start_ts

    def _compute_oval(h, w):
        cx = w // 2
        cy = int(h * cfg.GUIDE_OVAL_CY)
        ew = int(h * cfg.GUIDE_OVAL_EW)
        eh = int(h * cfg.GUIDE_OVAL_EH)
        return cx, cy, ew, eh

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            continue
        if cfg.CAMERA_FLIP:
            frame = cv2.flip(frame, 1)

        now    = datetime.now()
        now_ts = now.timestamp()
        frame_count += 1

        fps_counter += 1
        if now_ts - fps_timer >= 1.0:
            display_fps = fps_counter / (now_ts - fps_timer)
            fps_counter, fps_timer = 0, now_ts

        should_checkout = (
            (now_ts - start_ts >= cfg.TEST_DURATION_SECONDS) if cfg.TEST_MODE
            else (now.time() >= cfg.CHECKOUT_TIME)
        )
        if should_checkout and not checkout_done:
            checkout_done = True
            session.do_checkout(camera_name, now)
        if cfg.TEST_MODE and checkout_done:
            break

        do_detect = (frame_count % cfg.DETECT_EVERY_N_FRAMES == 0)
        if do_detect:
            last_faces = app.get(frame)

        hand_results = None
        has_challenge = any(
            lv.challenge_phase == "active"
            for lv in session.liveness.values()
            if not lv.confirmed and not lv.failed
        )
        if has_challenge:
            hand_results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        session.cleanup_expired(now_ts)

        if last_faces:
            last_face_ts = now_ts
        elif now_ts - last_face_ts >= cfg.NO_FACE_RESET_SEC:
            for name in list(session.liveness.keys()):
                person = session.persons.get(name)
                if not person or not person.checked_in:
                    del session.liveness[name]
            last_faces = []
            last_face_ts = now_ts

        _need_snapshot = any(
            not p.checked_in for p in session.persons.values()
        ) or not session.persons
        orig_frame = frame.copy() if _need_snapshot else frame

        fh, fw = frame.shape[:2]
        oval_cx, oval_cy, oval_ew, oval_eh = _compute_oval(fh, fw)

        _face_with_names = []

        for face in last_faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            left, top, right, bottom = x1, y1, x2, y2
            face_w   = right - left
            face_box = (top, right, bottom, left)

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

            embedding = face.embedding
            name = identify_face(embedding, known_norms, known_names)

            _face_with_names.append((face_box, name))

            pad  = 15
            crop = orig_frame[max(0, top-pad):min(fh, bottom+pad),
                              max(0, left-pad):min(fw, right+pad)]

            if name == "Unknown":
                ui.draw_face_box(frame, left, top, right, bottom,
                                 cfg.Color.UNKNOWN, "Unknown")
                continue

            lm_dict = None
            if face.landmark_3d_68 is not None:
                lm_dict = landmarks_68_to_dict(face.landmark_3d_68)

            person, liveness = session.get_or_create(name, now, crop)

            session.engine.update(
                liveness, lm_dict or {}, crop, face_box, face_w,
                frame, hand_results, now_ts, do_detect,
            )

            if liveness.confirmed and not person.checked_in:
                person.snapshot = orig_frame.copy()

            session.try_checkin(name, camera_name)

            if cfg.SHOW_LANDMARKS and lm_dict:
                ui.draw_landmarks(frame, lm_dict, scale=1.0)

            color, label = ui.get_face_visual(name, person, liveness)
            ui.draw_face_box(frame, left, top, right, bottom, color, label)

        ui.draw_face_guide(frame, _face_with_names,
                           session.liveness, session.persons, now_ts)
        ui.draw_hands(frame, hand_results)

        remaining = max(0, cfg.TEST_DURATION_SECONDS - int(now_ts - start_ts)) if cfg.TEST_MODE else 0
        ui.draw_hud(frame, display_fps, cfg.TEST_MODE, checkout_done, remaining)

        panel = ui.build_panel(session.persons, session.liveness, frame.shape[0])
        cv2.imshow(win_name, np.hstack([frame, panel]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    hands.close()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera(camera_index=0, camera_name="CAM_MAIN")
