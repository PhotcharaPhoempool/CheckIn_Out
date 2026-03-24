"""
ui_renderer.py — วาด UI ทั้งหมด (กรอบหน้า, panel, challenge overlay, HUD)
ทุก dimension อ่านจาก config.py
"""

import os
import cv2
import numpy as np
import mediapipe as mp
import config as cfg
from config import Color as C


# ─── Thai text rendering ───────────────────

_font_cache = {}
_font_path  = None
_font_searched = False

def _find_thai_font():
    global _font_path, _font_searched
    if _font_searched:
        return _font_path
    _font_searched = True
    import glob
    for pattern in [
        "/usr/share/fonts/truetype/thai/*.ttf",
        "/usr/share/fonts/truetype/tlwg/*.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansThai*.ttf",
        "/usr/share/fonts/opentype/noto/NotoSans*Thai*.otf",
        "/usr/share/fonts/truetype/noto/NotoSans*.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Thonburi.ttc",
        "C:/Windows/Fonts/tahoma.ttf",
        "fonts/*.ttf", "*.ttf",
    ]:
        for path in sorted(glob.glob(pattern)):
            if os.path.isfile(path):
                _font_path = path
                print(f"[FONT] ใช้ฟอนต์: {path}")
                return _font_path
    try:
        import subprocess
        result = subprocess.run(["fc-list", ":lang=th", "file"],
                                capture_output=True, text=True, timeout=3)
        for line in result.stdout.strip().split("\n"):
            path = line.strip().rstrip(":")
            if path and os.path.isfile(path):
                _font_path = path
                print(f"[FONT] ใช้ฟอนต์ (fc-list): {path}")
                return _font_path
    except Exception:
        pass
    print("[FONT] WARNING: ไม่พบฟอนต์ — sudo apt install fonts-tlwg-garuda")
    return None

def _get_font(size: int):
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
    font = _get_font(max(10, int(scale * 28)))
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
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


# ─── Face box + label (ใช้ cfg.FACEBOX_*) ──

def draw_face_box(frame, left, top, right, bottom, color, label):
    cv2.rectangle(frame, (left, top), (right, bottom), color, cfg.FACEBOX_THICK)
    lh = cfg.FACEBOX_LABEL_H
    cv2.rectangle(frame, (left, bottom - lh), (right, bottom), color, cv2.FILLED)
    cv2.putText(frame, label, (left + cfg.FACEBOX_PAD, bottom - 8),
                cv2.FONT_HERSHEY_SIMPLEX, cfg.FACEBOX_FONT, C.TEXT, 1)


def draw_landmarks(frame, lm_dict, scale=1.0):
    for group in ["nose_bridge", "nose_tip", "chin"]:
        for px, py in lm_dict.get(group, []):
            cv2.circle(frame, (int(px / scale), int(py / scale)), 2, C.LIVENESS, -1)


def draw_hands(frame, hand_results):
    if hand_results and hand_results.multi_hand_landmarks:
        for hlm in hand_results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hlm, mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style(),
            )


# ─── Face color + label ─────────────────────

def get_face_visual(name, person, liveness):
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


# ─── Challenge overlay ──────────────────────

def draw_challenge_overlay(frame, liveness, now_ts):
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
    det = liveness.challenge_detected
    if det >= 0:
        cv2.putText(frame, f"Detected: {det}", (tx, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    C.OK if det == cn else C.FAIL, 2)


# ─── Face Guide Oval (ใช้ cfg.GUIDE_*) ──────

def draw_face_guide(frame, face_boxes_full: list, liveness_map: dict,
                    persons: dict = None, now_ts: float = 0.0):
    if not cfg.GUIDE_OVERLAY:
        return

    h, w = frame.shape[:2]
    cx = w // 2
    cy = int(h * cfg.GUIDE_OVAL_CY)
    ew = int(h * cfg.GUIDE_OVAL_EW)
    eh = int(h * cfg.GUIDE_OVAL_EH)

    # Dim พื้นที่นอกวงรี
    dark = (frame.astype(np.float32) * cfg.GUIDE_DIM_FACTOR).astype(np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (ew, eh), 0, 0, 360, 255, -1)
    blur = cv2.GaussianBlur(mask.astype(np.float32),
                             (cfg.GUIDE_DIM_BLUR, cfg.GUIDE_DIM_BLUR), 0) / 255.0
    alpha = blur[..., np.newaxis]
    frame[:] = (frame * alpha + dark * (1.0 - alpha)).astype(np.uint8)

    # หาหน้าในวงรี + ชื่อ
    face_in_oval = False
    oval_name = None
    for item in face_boxes_full:
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], tuple):
            face_box, name = item
        else:
            face_box, name = item, None
        top, right, bottom, left = face_box
        fcx = (left + right) / 2.0
        fcy = (top + bottom) / 2.0
        if ((fcx - cx) / ew) ** 2 + ((fcy - cy) / eh) ** 2 <= cfg.GUIDE_IN_OVAL_TOL:
            face_in_oval, oval_name = True, name
            break

    # สถานะ — เฉพาะเมื่อมีหน้าจริงในวงรี
    oval_color = C.OVAL_DEFAULT
    main_text  = "Place your face in the oval"
    sub_text   = ""

    if face_in_oval and oval_name and oval_name in (liveness_map or {}):
        lv = liveness_map[oval_name]
        person = (persons or {}).get(oval_name)

        if (person and person.checked_in) or lv.confirmed:
            oval_color, main_text = C.CHECKED_IN, "Verified!"
        elif lv.failed:
            oval_color = C.LIVENESS_FAIL
            retry_at = lv.start_ts + cfg.LIVENESS_TIMEOUT + cfg.LIVENESS_RETRY_AFTER
            main_text = "Verification failed"
            sub_text = f"{_format_fail_reason(lv.fail_reason)}  |  Retry {max(0,retry_at-now_ts):.0f}s"
        elif lv.challenge_phase == "active":
            cn, dn = lv.challenge_number, len(lv.challenge_done)
            rem = max(0, cfg.CHALLENGE_TIMEOUT - (now_ts - lv.challenge_start_ts))
            oval_color = C.CHALLENGE
            main_text = f"Show {cn} finger(s)"
            sub_text = f"Step {dn+1}/{cfg.CHALLENGE_COUNT}  |  {rem:.1f}s left"
        else:
            dp = lv.depth_pass_count
            m = "OK" if lv.motion_ok else "--"
            t = "OK" if lv.texture_ok else "--"
            s = "OK" if lv.screen_ok else "FAIL"
            f = "OK" if lv.fas_ok else (f"{lv.fas_score:.1f}" if lv.fas_score >= 0 else "--")
            oval_color = C.LIVENESS
            main_text = f"Hold still...  {dp}/{cfg.DEPTH_FRAMES_REQUIRED}"
            sub_text = f"Motion {m}  Tex {t}  Scr {s}  FAS {f}"
    elif face_in_oval:
        oval_color, main_text = C.LIVENESS, "Hold still..."

    # วงรี
    cv2.ellipse(frame, (cx, cy), (ew, eh), 0, 0, 360, oval_color, cfg.GUIDE_OVAL_THICK)
    inner = tuple(min(255, int(c * 1.25)) for c in oval_color)
    cv2.ellipse(frame, (cx, cy), (ew-5, eh-5), 0, 0, 360, inner, cfg.GUIDE_OVAL_INNER)

    _draw_instruction_card(frame, main_text, sub_text, oval_color)


def _format_fail_reason(reason: str) -> str:
    labels = {"Depth": "Face depth", "Motion": "No movement", "Texture": "Texture",
              "Screen": "Screen detected", "Finger": "Finger", "FAS": "AI spoof detected"}
    if reason.startswith("FAIL:"):
        return " + ".join(labels.get(p, p) for p in reason[5:].split("+"))
    if reason.startswith("Screen"):
        return "Phone/screen detected"
    if reason.startswith("Finger"):
        return f"Wrong fingers (expected {reason.split(':')[1] if ':' in reason else '?'})"
    if reason.startswith("FAS:"):
        return f"AI spoof ({reason.split('(')[1].rstrip(')') if '(' in reason else '?'})"
    return reason or "Unknown"


# ─── Instruction Card (ใช้ cfg.CARD_*) ──────

def _draw_instruction_card(frame, main_text: str, sub_text: str, accent: tuple):
    h, w = frame.shape[:2]
    ch = cfg.CARD_HEIGHT if sub_text else cfg.CARD_HEIGHT_SMALL
    cw = min(w - 40, cfg.CARD_WIDTH_MAX)
    cx = (w - cw) // 2
    cy = h - ch - cfg.CARD_MARGIN_BOTTOM

    # เงา
    shadow = frame.copy()
    so = cfg.CARD_SHADOW_OFF
    cv2.rectangle(shadow, (cx+so, cy+so), (cx+cw+so, cy+ch+so), (0,0,0), -1)
    frame[:] = cv2.addWeighted(frame, 1.0, shadow, cfg.CARD_SHADOW_ALPHA, 0)

    # พื้น + accent + กรอบ
    cv2.rectangle(frame, (cx, cy), (cx+cw, cy+ch), cfg.CARD_BG, -1)
    cv2.rectangle(frame, (cx, cy), (cx+cfg.CARD_ACCENT_W, cy+ch), accent, -1)
    cv2.rectangle(frame, (cx, cy), (cx+cw, cy+ch), accent, 2)

    # ข้อความ
    pl = cfg.CARD_PADDING_LEFT
    draw_text(frame, main_text, (cx+pl, cy+cfg.CARD_MAIN_Y),
              scale=cfg.CARD_FONT_MAIN, color=cfg.CARD_TEXT_MAIN)
    if sub_text:
        draw_text(frame, sub_text, (cx+pl, cy+cfg.CARD_SUB_Y),
                  scale=cfg.CARD_FONT_SUB, color=cfg.CARD_TEXT_SUB)


# ─── HUD ─────────────────────────────────────

def draw_hud(frame, fps, test_mode, checkout_done, remaining_sec=0):
    h = frame.shape[0]
    cv2.putText(frame, f"FPS:{fps:.0f}", (10, h-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, C.TEXT_DIM, 1)
    if test_mode and not checkout_done:
        cv2.putText(frame, f"[TEST] OUT in {remaining_sec}s", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, C.HUD_TEST, 2)
    elif checkout_done:
        cv2.putText(frame, "Session Ended", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, C.HUD_DONE, 2)


# ─── Side Panel (ใช้ cfg.PANEL_*) ───────────

def build_panel(persons: dict, liveness_map: dict, frame_height: int) -> np.ndarray:
    W = cfg.PANEL_WIDTH
    panel = np.zeros((frame_height, W, 3), dtype=np.uint8)
    panel[:] = C.PANEL_BG

    cv2.rectangle(panel, (0, 0), (W, cfg.PANEL_HEADER_H), C.PANEL_HEADER, cv2.FILLED)
    cv2.putText(panel, "Detected Persons", (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, C.TEXT, 1)

    FS = cfg.PANEL_FACE_SIZE
    ITEM_H = FS + cfg.PANEL_ITEM_EXTRA
    y = cfg.PANEL_START_Y

    for name, person in list(persons.items()):
        if y + ITEM_H > frame_height - 5:
            remaining = len(persons) - list(persons.keys()).index(name)
            cv2.putText(panel, f"+ {remaining} more...", (8, y+14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, C.TEXT_MORE, 1)
            break

        lv = liveness_map.get(name)
        color = get_face_visual(name, person, lv)[0] if lv else C.NOT_IN_DB

        xf = cfg.PANEL_FACE_X
        snap = person.snapshot
        if snap is not None and snap.size > 0:
            try:
                panel[y:y+FS, xf:xf+FS] = cv2.resize(snap, (FS, FS))
            except Exception:
                cv2.rectangle(panel, (xf, y), (xf+FS, y+FS), C.FACE_PH, cv2.FILLED)
        else:
            cv2.rectangle(panel, (xf, y), (xf+FS, y+FS), C.FACE_PH, cv2.FILLED)
        cv2.rectangle(panel, (xf, y), (xf+FS, y+FS), color, 2)

        tx = xf + FS + 8
        draw_text(panel, name[:14], (tx, y+14), scale=0.40, color=C.TEXT_CYAN)
        if person.first_seen:
            cv2.putText(panel, f"IN: {person.first_seen.strftime('%H:%M:%S')}",
                        (tx, y+FS-28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, C.CHECKED_IN, 1)
        if person.last_seen:
            cv2.putText(panel, f"LST:{person.last_seen.strftime('%H:%M:%S')}",
                        (tx, y+FS-14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, C.TEXT_DIM, 1)

        if lv:
            _, sl = get_face_visual(name, person, lv)
            sl = sl.replace(f"{name} ", "").strip("[]")
        else:
            sl = "N/A"
        cv2.putText(panel, sl[:30], (xf, y+FS+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1)

        y += ITEM_H
        cv2.line(panel, (5, y-4), (W-5, y-4), C.DIVIDER, 1)

    return panel