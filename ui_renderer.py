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

_thai_font = None
_thai_font_loaded = False

def _load_thai_font(size=12):
    """โหลดฟอนต์ไทย ครั้งเดียว"""
    global _thai_font, _thai_font_loaded
    if _thai_font_loaded:
        return _thai_font
    _thai_font_loaded = True
    try:
        from PIL import ImageFont
        for path in [
            "/usr/share/fonts/truetype/thai/Garuda.ttf",
            "/usr/share/fonts/truetype/tlwg/Garuda.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansThai-Regular.otf",
        ]:
            if os.path.exists(path):
                _thai_font = path
                return _thai_font
    except ImportError:
        pass
    return None


def draw_text(img, text, pos, scale=0.45, color=C.TEXT, thickness=1):
    """วาดข้อความ (รองรับภาษาไทย ถ้ามี PIL)"""
    font_path = _load_thai_font()
    if font_path:
        try:
            from PIL import ImageFont, ImageDraw, Image
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            font = ImageFont.truetype(font_path, max(10, int(scale * 28)))
            draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
            img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            return
        except Exception:
            pass
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