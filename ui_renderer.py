"""
ui_renderer.py — วาด UI ทั้งหมด (กรอบหน้า, panel, challenge overlay, HUD)
"""

import os
import cv2
import numpy as np
import mediapipe as mp
import config as cfg
from config import Color as C


# ─── Thai text rendering ───────────────────

_font_cache = {}       # {size: PIL.ImageFont}
_font_path  = None     # path ที่หาได้
_font_searched = False

def _find_thai_font():
    """หาฟอนต์ที่รองรับภาษาไทย — ค้นหาทุกที่ที่เป็นไปได้"""
    global _font_path, _font_searched
    if _font_searched:
        return _font_path
    _font_searched = True

    import glob

    # 1) ลองจาก path ที่รู้จัก
    known_paths = [
        # TH-specific
        "/usr/share/fonts/truetype/thai/*.ttf",
        "/usr/share/fonts/truetype/tlwg/*.ttf",
        # Noto (รองรับ Thai)
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansThai*.ttf",
        "/usr/share/fonts/opentype/noto/NotoSans*Thai*.otf",
        "/usr/share/fonts/truetype/noto/NotoSans*.ttf",
        # DejaVu / Liberation (fallback)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        # macOS
        "/System/Library/Fonts/Supplemental/Thonburi.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        # Windows
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/arial.ttf",
        # User project fonts
        "fonts/*.ttf",
        "*.ttf",
    ]

    for pattern in known_paths:
        matches = glob.glob(pattern)
        for path in sorted(matches):
            if os.path.isfile(path):
                _font_path = path
                print(f"[FONT] ใช้ฟอนต์: {path}")
                return _font_path

    # 2) ลองใช้ fc-list หาฟอนต์ Thai
    try:
        import subprocess
        result = subprocess.run(
            ["fc-list", ":lang=th", "file"],
            capture_output=True, text=True, timeout=3,
        )
        for line in result.stdout.strip().split("\n"):
            path = line.strip().rstrip(":")
            if path and os.path.isfile(path):
                _font_path = path
                print(f"[FONT] ใช้ฟอนต์ (fc-list): {path}")
                return _font_path
    except Exception:
        pass

    # 3) หาอะไรก็ได้ที่เป็น .ttf
    try:
        import subprocess
        result = subprocess.run(
            ["fc-list", "", "file"],
            capture_output=True, text=True, timeout=3,
        )
        for line in result.stdout.strip().split("\n"):
            path = line.strip().rstrip(":")
            if path and path.endswith((".ttf", ".otf")) and os.path.isfile(path):
                _font_path = path
                print(f"[FONT] ใช้ฟอนต์ (fallback): {path}")
                return _font_path
    except Exception:
        pass

    print("[FONT] WARNING: ไม่พบฟอนต์ — ภาษาไทยจะแสดงไม่ได้")
    print("[FONT] แก้ไข: sudo apt install fonts-tlwg-garuda")
    return None


def _get_font(size: int):
    """โหลด PIL font ตาม size (cache ไว้)"""
    if size in _font_cache:
        return _font_cache[size]

    path = _find_thai_font()
    if not path:
        return None

    try:
        from PIL import ImageFont
        font = ImageFont.truetype(path, size)
        _font_cache[size] = font
        return font
    except Exception:
        return None


def draw_text(img, text, pos, scale=0.45, color=C.TEXT, thickness=1):
    """วาดข้อความ (รองรับภาษาไทย ถ้ามี PIL + font)"""
    font_size = max(10, int(scale * 28))
    font = _get_font(font_size)

    if font:
        try:
            from PIL import ImageDraw, Image
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
            img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            return
        except Exception:
            pass

    # Fallback: OpenCV (ภาษาไทยจะเป็นกล่อง แต่ภาษาอังกฤษ OK)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


# ─── Face box + label ──────────────────────

def draw_face_box(frame, left, top, right, bottom, color, label):
    """วาดกรอบหน้า + label ด้านล่าง"""
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    cv2.rectangle(frame, (left, bottom - 28), (right, bottom), color, cv2.FILLED)
    cv2.putText(frame, label, (left + 5, bottom - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, C.TEXT, 1)


def draw_landmarks(frame, lm_dict, scale):
    """วาดจุด landmark (จมูก + คาง)"""
    for group in ["nose_bridge", "nose_tip", "chin"]:
        for px, py in lm_dict.get(group, []):
            cv2.circle(frame, (int(px / scale), int(py / scale)), 2, C.LIVENESS, -1)


def draw_hands(frame, hand_results):
    """วาด hand landmarks (MediaPipe)"""
    if hand_results and hand_results.multi_hand_landmarks:
        for hlm in hand_results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hlm, mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style(),
            )


# ─── Determine face color + label ─────────

def get_face_visual(name, person, liveness):
    """
    คืน (color, label) ตามสถานะ liveness + check-in
    """
    if liveness.failed:
        return C.LIVENESS_FAIL, f"{name} [{liveness.fail_reason}]"

    if liveness.challenge_phase == "active" and not liveness.confirmed:
        cn = liveness.challenge_number
        dn = len(liveness.challenge_done)
        det = liveness.challenge_detected
        det_str = f" see:{det}" if det >= 0 else ""
        return C.CHALLENGE, f"{name} [Show {cn} ({dn+1}/{cfg.CHALLENGE_COUNT}){det_str}]"

    if not liveness.confirmed:
        dp = liveness.depth_pass_count
        return C.LIVENESS, f"{name} [Scan {dp}/{cfg.DEPTH_FRAMES_REQUIRED}]"

    if person.checked_out:
        return C.CHECKED_OUT, name
    if person.checked_in:
        return C.CHECKED_IN, name
    return C.NOT_IN_DB, name


# ─── Challenge overlay (กลางจอ) ────────────

def draw_challenge_overlay(frame, liveness, now_ts):
    """วาดคำสั่ง challenge ตัวใหญ่กลางจอ"""
    if liveness.challenge_phase != "active" or liveness.confirmed or liveness.failed:
        return

    cn  = liveness.challenge_number
    dn  = len(liveness.challenge_done)
    rem = max(0, cfg.CHALLENGE_TIMEOUT - (now_ts - liveness.challenge_start_ts))

    txt = f"Show {cn} finger(s)"
    sz  = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    tx  = (frame.shape[1] - sz[0]) // 2

    cv2.putText(frame, txt, (tx, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, C.CHALLENGE, 3)
    cv2.putText(frame, f"({dn+1}/{cfg.CHALLENGE_COUNT})  {rem:.1f}s",
                (tx, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C.TEXT, 1)

    # จำนวนนิ้วที่ detect
    det = liveness.challenge_detected
    if det >= 0:
        det_c = C.OK if det == cn else C.FAIL
        cv2.putText(frame, f"Detected: {det}", (tx, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, det_c, 2)


# ─── Face Guide Oval Overlay ──────────────

def draw_face_guide(frame, face_boxes_full: list, liveness_map: dict,
                    persons: dict = None, now_ts: float = 0.0):
    """
    วาด oval guide ช่วยไกด์ตำแหน่งหน้า + dim พื้นที่นอกวงรี
    face_boxes_full : list of (top, right, bottom, left) ใน full-frame px
    liveness_map    : {name: LivenessState}
    now_ts          : timestamp ปัจจุบัน (สำหรับ countdown retry)
    """
    if not cfg.GUIDE_OVERLAY:
        return

    h, w = frame.shape[:2]
    cx  = w // 2
    cy  = int(h * 0.47)
    ew  = int(h * 0.29)   # แกน x วงรี
    eh  = int(h * 0.34)   # แกน y วงรี (ลดจาก 0.39)

    # ── Dim พื้นที่นอกวงรี ──
    dark = (frame.astype(np.float32) * 0.38).astype(np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (ew, eh), 0, 0, 360, 255, -1)
    blur = cv2.GaussianBlur(mask.astype(np.float32), (41, 41), 0) / 255.0
    alpha = blur[..., np.newaxis]
    frame[:] = (frame * alpha + dark * (1.0 - alpha)).astype(np.uint8)

    # ── ตรวจว่ามีหน้าอยู่ในวงรีหรือเปล่า + ดึงชื่อ ──
    face_in_oval = False
    oval_name = None
    for item in face_boxes_full:
        # รองรับทั้ง (face_box, name) และ face_box โดยตรง
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], tuple):
            face_box, name = item
        else:
            face_box = item
            name = None
        top, right, bottom, left = face_box
        fcx = (left + right) / 2.0
        fcy = (top  + bottom) / 2.0
        if ((fcx - cx) / ew) ** 2 + ((fcy - cy) / eh) ** 2 <= 1.35:
            face_in_oval = True
            oval_name = name
            break

    # ── สีและข้อความตามสถานะ — เฉพาะเมื่อมีหน้าจริงในวงรี ──
    oval_color  = (210, 210, 210)
    main_text   = "Place your face in the oval"
    sub_text    = ""

    if face_in_oval and oval_name and oval_name in (liveness_map or {}):
        lv = liveness_map[oval_name]
        active_person = (persons or {}).get(oval_name)

        if (active_person and active_person.checked_in) or lv.confirmed:
            oval_color = C.CHECKED_IN
            main_text  = "Verified!"
            sub_text   = ""
        elif lv.failed:
            oval_color = C.LIVENESS_FAIL
            retry_at   = lv.start_ts + cfg.LIVENESS_TIMEOUT + cfg.LIVENESS_RETRY_AFTER
            remaining  = max(0.0, retry_at - now_ts)
            main_text  = "Verification failed"
            sub_text   = f"{_format_fail_reason(lv.fail_reason)}  |  Retry in {remaining:.0f}s"
        elif lv.challenge_phase == "active":
            cn         = lv.challenge_number
            dn         = len(lv.challenge_done)
            rem        = max(0.0, cfg.CHALLENGE_TIMEOUT - (now_ts - lv.challenge_start_ts))
            oval_color = C.CHALLENGE
            main_text  = f"Show {cn} finger(s)"
            sub_text   = f"Step {dn + 1}/{cfg.CHALLENGE_COUNT}  |  {rem:.1f}s left"
        else:
            dp         = lv.depth_pass_count
            total      = cfg.DEPTH_FRAMES_REQUIRED
            m_ok       = "OK" if lv.motion_ok  else "--"
            t_ok       = "OK" if lv.texture_ok else "--"
            s_ok       = "OK" if lv.screen_ok  else "FAIL"
            f_ok       = "OK" if lv.fas_ok     else (f"{lv.fas_score:.1f}" if lv.fas_score >= 0 else "--")
            oval_color = C.LIVENESS
            main_text  = f"Hold still...  {dp}/{total}"
            sub_text   = f"Motion {m_ok}  Tex {t_ok}  Scr {s_ok}  FAS {f_ok}"
    elif face_in_oval:
        oval_color = C.LIVENESS
        main_text  = "Hold still..."
        sub_text   = ""

    # ── วาดวงรี 2 ชั้น (outer bold + inner thin) ──
    cv2.ellipse(frame, (cx, cy), (ew, eh),         0, 0, 360, oval_color, 3)
    inner = tuple(min(255, int(c * 1.25)) for c in oval_color)
    cv2.ellipse(frame, (cx, cy), (ew - 5, eh - 5), 0, 0, 360, inner,      1)

    # ── Instruction card ด้านล่าง ──
    _draw_instruction_card(frame, main_text, sub_text, oval_color)


def _format_fail_reason(reason: str) -> str:
    """แปล fail_reason ให้อ่านง่าย"""
    _labels = {"Depth": "Face depth", "Motion": "No movement",
               "Texture": "Texture", "Screen": "Screen detected",
               "Finger": "Finger", "FAS": "AI spoof detected"}
    if reason.startswith("FAIL:"):
        parts = reason[5:].split("+")
        return " + ".join(_labels.get(p, p) for p in parts)
    if reason.startswith("Screen"):
        return "Phone/screen detected"
    if reason.startswith("Finger"):
        num = reason.split(":")[1] if ":" in reason else "?"
        return f"Wrong fingers (expected {num})"
    if reason.startswith("FAS:"):
        score = reason.split("(")[1].rstrip(")") if "(" in reason else "?"
        return f"AI detected spoof (score {score})"
    return reason or "Unknown"


def _draw_instruction_card(frame, main_text: str, sub_text: str, accent: tuple):
    """การ์ดขาวด้านล่างแสดงคำแนะนำ 2 บรรทัด"""
    h, w    = frame.shape[:2]
    CARD_H  = 82 if sub_text else 58
    CARD_W  = min(w - 40, 420)
    cx_card = (w - CARD_W) // 2
    cy_card = h - CARD_H - 14   # ด้านล่าง

    # เงา
    shadow = frame.copy()
    cv2.rectangle(shadow,
                  (cx_card + 3, cy_card + 3),
                  (cx_card + CARD_W + 3, cy_card + CARD_H + 3),
                  (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(frame, 1.0, shadow, 0.35, 0)

    # พื้นหลังขาว
    cv2.rectangle(frame,
                  (cx_card, cy_card),
                  (cx_card + CARD_W, cy_card + CARD_H),
                  (245, 245, 245), -1)

    # แถบสีบนบัตร (accent bar ด้านซ้าย)
    cv2.rectangle(frame,
                  (cx_card, cy_card),
                  (cx_card + 6, cy_card + CARD_H),
                  accent, -1)

    # กรอบ
    cv2.rectangle(frame,
                  (cx_card, cy_card),
                  (cx_card + CARD_W, cy_card + CARD_H),
                  accent, 2)

    # บรรทัดหลัก
    draw_text(frame, main_text,
              (cx_card + 18, cy_card + 14),
              scale=0.72,
              color=(25, 25, 25))

    # บรรทัดรอง (ถ้ามี)
    if sub_text:
        draw_text(frame, sub_text,
                  (cx_card + 18, cy_card + 50),
                  scale=0.50,
                  color=(80, 80, 80))


# ─── HUD (FPS, test timer) ────────────────

def draw_hud(frame, fps, test_mode, checkout_done, remaining_sec=0):
    """วาด FPS + สถานะ test/checkout"""
    h = frame.shape[0]
    cv2.putText(frame, f"FPS:{fps:.0f}", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, C.TEXT_DIM, 1)

    if test_mode and not checkout_done:
        cv2.putText(frame, f"[TEST] OUT in {remaining_sec}s", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, C.HUD_TEST, 2)
    elif checkout_done:
        cv2.putText(frame, "Session Ended", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, C.HUD_DONE, 2)


# ─── Side panel ───────────────────────────

def build_panel(persons: dict, liveness_map: dict, frame_height: int) -> np.ndarray:
    """สร้าง panel ด้านขวา แสดงรายชื่อ + สถานะ"""
    W = cfg.PANEL_WIDTH
    panel = np.zeros((frame_height, W, 3), dtype=np.uint8)
    panel[:] = C.PANEL_BG

    # Header
    cv2.rectangle(panel, (0, 0), (W, 38), C.PANEL_HEADER, cv2.FILLED)
    cv2.putText(panel, "Detected Persons", (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, C.TEXT, 1)

    FACE_W, FACE_H = 80, 80
    ITEM_H = FACE_H + 58
    y = 46

    for name, person in list(persons.items()):
        if y + ITEM_H > frame_height - 5:
            remaining = len(persons) - list(persons.keys()).index(name)
            cv2.putText(panel, f"+ {remaining} more...", (8, y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, C.TEXT_MORE, 1)
            break

        lv = liveness_map.get(name)
        color, _ = get_face_visual(name, person, lv) if lv else (C.NOT_IN_DB, name)

        # Snapshot
        xf = 8
        snap = person.snapshot
        if snap is not None and snap.size > 0:
            try:
                panel[y:y+FACE_H, xf:xf+FACE_W] = cv2.resize(snap, (FACE_W, FACE_H))
            except Exception:
                cv2.rectangle(panel, (xf, y), (xf+FACE_W, y+FACE_H), C.FACE_PH, cv2.FILLED)
        else:
            cv2.rectangle(panel, (xf, y), (xf+FACE_W, y+FACE_H), C.FACE_PH, cv2.FILLED)
        cv2.rectangle(panel, (xf, y), (xf+FACE_W, y+FACE_H), color, 2)

        # Name + times
        tx = xf + FACE_W + 8
        draw_text(panel, name[:14], (tx, y + 14), scale=0.40, color=C.TEXT_CYAN)

        if person.first_seen:
            cv2.putText(panel, f"IN: {person.first_seen.strftime('%H:%M:%S')}",
                        (tx, y + FACE_H - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, C.CHECKED_IN, 1)
        if person.last_seen:
            cv2.putText(panel, f"LST:{person.last_seen.strftime('%H:%M:%S')}",
                        (tx, y + FACE_H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, C.TEXT_DIM, 1)

        # Status label
        if lv:
            _, status_label = get_face_visual(name, person, lv)
            status_label = status_label.replace(f"{name} ", "").strip("[]")
        else:
            status_label = "N/A"
        cv2.putText(panel, status_label[:30], (xf, y + FACE_H + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1)

        y += ITEM_H
        cv2.line(panel, (5, y - 4), (W - 5, y - 4), C.DIVIDER, 1)

    return panel