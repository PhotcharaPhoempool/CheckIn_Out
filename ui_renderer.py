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
    fh, fw = frame.shape[:2]
    y1 = max(0, bottom - lh)
    y2 = min(fh, bottom)
    x1 = max(0, left)
    x2 = min(fw, right)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FILLED)
    # render text บน label ROI เท่านั้น (ไม่แปลง full frame เป็น PIL)
    if x2 > x1 and y2 > y1:
        roi = frame[y1:y2, x1:x2]
        draw_text(roi, label, (cfg.FACEBOX_PAD, 5), scale=cfg.FACEBOX_FONT, color=C.TEXT)
        frame[y1:y2, x1:x2] = roi


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
    display = (getattr(person, "display_name", None) or name)[:24]
    if liveness.failed:
        return C.LIVENESS_FAIL, f"{display} [{liveness.fail_reason}]"
    if liveness.challenge_phase == "active" and not liveness.confirmed:
        cn = liveness.challenge_number
        dn = len(liveness.challenge_done)
        return C.CHALLENGE, f"{display} [Show {cn} ({dn+1}/{cfg.CHALLENGE_COUNT})]"
    if not liveness.confirmed:
        # แสดงด่านที่กำลัง block อยู่
        dp = liveness.depth_pass_count
        if dp < cfg.DEPTH_FRAMES_REQUIRED:
            tag = f"Scan {dp}/{cfg.DEPTH_FRAMES_REQUIRED}"
        elif cfg.BLINK_ENABLED and not liveness.blink_ok:
            tag = f"Blink {liveness.blink_count}/{cfg.BLINK_MIN}"
        elif not liveness.motion_ok:
            tag = "Move"
        elif not liveness.texture_ok:
            tag = "Texture"
        elif not liveness.fas_ok:
            tag = "AI"
        else:
            tag = "Wait"
        return C.LIVENESS, f"{display} [{tag}]"
    if person.checked_out:
        return C.CHECKED_OUT, display
    if person.checked_in:
        return C.CHECKED_IN, display
    return C.NOT_IN_DB, display


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

_oval_alpha_cache: dict = {}   # cache alpha mask เพื่อหลีกเลี่ยง GaussianBlur ทุก frame

def _get_oval_blend(h, w, cx, cy, ew, eh):
    """
    คืน blend_u8_3ch uint8 (h,w,3) — คำนวณครั้งเดียวต่อ frame size
    ใช้: frame[:] = ((frame.astype(uint16) * blend_u8_3ch) >> 8).astype(uint8)
    เร็วกว่า float32 blend ~10x
    """
    key = (h, w, cx, cy, ew, eh, cfg.GUIDE_DIM_FACTOR)
    if key not in _oval_alpha_cache:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (ew, eh), 0, 0, 360, 255, -1)
        blur = cv2.GaussianBlur(mask.astype(np.float32),
                                (cfg.GUIDE_DIM_BLUR, cfg.GUIDE_DIM_BLUR), 0) / 255.0
        dim = cfg.GUIDE_DIM_FACTOR
        blend_f = (blur * (1.0 - dim) + dim)
        # uint8 3-channel ให้ทำ integer multiply ได้เลย (เร็วกว่า float ~10x)
        blend_u8 = np.repeat((blend_f * 255).astype(np.uint8)[..., np.newaxis], 3, axis=2)
        _oval_alpha_cache[key] = blend_u8
        if len(_oval_alpha_cache) > 4:
            _oval_alpha_cache.pop(next(iter(_oval_alpha_cache)))
    return _oval_alpha_cache[key]


def draw_face_guide(frame, face_boxes_full: list, liveness_map: dict,
                    persons: dict = None, now_ts: float = 0.0):
    if not cfg.GUIDE_OVERLAY or cfg.TEST_MODE:
        return

    h, w = frame.shape[:2]
    cx = w // 2
    cy = int(h * cfg.GUIDE_OVAL_CY)
    ew = int(h * cfg.GUIDE_OVAL_EW)
    eh = int(h * cfg.GUIDE_OVAL_EH)

    # Dim พื้นที่นอกวงรี — uint16 integer multiply (เร็ว ~10x vs float32)
    blend = _get_oval_blend(h, w, cx, cy, ew, eh)
    frame[:] = ((frame.astype(np.uint16) * blend) >> 8).astype(np.uint8)

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
            m  = "OK" if lv.motion_ok  else "--"
            bk = f"{lv.blink_count}/{cfg.BLINK_MIN}" if not lv.blink_ok else "OK"
            t  = "OK" if lv.texture_ok else "--"
            s  = "OK" if lv.screen_ok  else "FAIL"
            f  = "OK" if lv.fas_ok     else (f"{lv.fas_score:.1f}" if lv.fas_score >= 0 else "--")
            oval_color = C.LIVENESS

            # แสดงว่ากำลังรอด่านไหน
            if dp < cfg.DEPTH_FRAMES_REQUIRED:
                main_text = f"Hold still...  {dp}/{cfg.DEPTH_FRAMES_REQUIRED}"
            elif cfg.BLINK_ENABLED and not lv.blink_ok:
                main_text = f"Please blink  {lv.blink_count}/{cfg.BLINK_MIN}"
            elif not lv.motion_ok:
                main_text = "Move your head slightly"
            elif not lv.texture_ok:
                main_text = "Hold still (texture)..."
            elif not lv.fas_ok:
                main_text = "Hold still (AI check)..."
            else:
                main_text = "Hold still..."

            bk_part  = f"  Blink {bk}" if cfg.BLINK_ENABLED else ""
            scr_part = f"  Scr {s}" if cfg.SCREEN_DETECT_ENABLED else ""
            sub_text = f"Mot {m}{bk_part}  Tex {t}{scr_part}  FAS {f}"
    elif face_in_oval:
        oval_color, main_text = C.LIVENESS, "Hold still..."

    # วงรี
    cv2.ellipse(frame, (cx, cy), (ew, eh), 0, 0, 360, oval_color, cfg.GUIDE_OVAL_THICK)
    inner = tuple(min(255, int(c * 1.25)) for c in oval_color)
    cv2.ellipse(frame, (cx, cy), (ew-5, eh-5), 0, 0, 360, inner, cfg.GUIDE_OVAL_INNER)

    _draw_instruction_card(frame, main_text, sub_text, oval_color)


def _format_fail_reason(reason: str) -> str:
    labels = {"Depth": "Face depth", "Motion": "No movement", "Blink": "No blink",
              "Texture": "Texture", "Screen": "Screen detected",
              "Finger": "Finger", "FAS": "AI spoof detected"}
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

    # เงา — ดำเฉพาะ shadow region (ไม่ addWeighted ทั้ง frame)
    so = cfg.CARD_SHADOW_OFF
    sy1 = max(0, cy + so)
    sy2 = min(h, cy + ch + so)
    sx1 = max(0, cx + so)
    sx2 = min(w, cx + cw + so)
    frame[sy1:sy2, sx1:sx2] = (
        frame[sy1:sy2, sx1:sx2].astype(np.float32) * (1.0 - cfg.CARD_SHADOW_ALPHA)
    ).astype(np.uint8)

    # พื้น + accent + กรอบ
    cv2.rectangle(frame, (cx, cy), (cx+cw, cy+ch), cfg.CARD_BG, -1)
    cv2.rectangle(frame, (cx, cy), (cx+cfg.CARD_ACCENT_W, cy+ch), accent, -1)
    cv2.rectangle(frame, (cx, cy), (cx+cw, cy+ch), accent, 2)

    # ข้อความ — ทำงานบน card ROI เท่านั้น (ไม่แปลง full frame เป็น PIL)
    card_roi = frame[cy:cy+ch, cx:cx+cw]
    pl = cfg.CARD_PADDING_LEFT
    draw_text(card_roi, main_text, (pl, cfg.CARD_MAIN_Y),
              scale=cfg.CARD_FONT_MAIN, color=cfg.CARD_TEXT_MAIN)
    if sub_text:
        draw_text(card_roi, sub_text, (pl, cfg.CARD_SUB_Y),
                  scale=cfg.CARD_FONT_SUB, color=cfg.CARD_TEXT_SUB)
    frame[cy:cy+ch, cx:cx+cw] = card_roi


# ─── HUD ─────────────────────────────────────

def draw_screen_debug(frame, dbg: dict):
    """
    แสดงข้อมูล Screen Border Detection แบบ real-time สำหรับ calibration
    เรียกเฉพาะตอน TEST_MODE=True — ไม่กระทบ production

    สี:
      เขียว  = เส้นแนวนอนที่อยู่ใน edge zone → นับเป็น edge_H
      ฟ้า    = เส้นแนวตั้งที่อยู่ใน edge zone → นับเป็น edge_V
      เทา    = เส้นอื่น (นอก zone หรือ diagonal) → ไม่นับ
      กรอบส้ม = margin zone (ROI ที่สแกน)
    """
    if not dbg or not dbg.get("valid"):
        return

    roi_y1, roi_x1, roi_y2, roi_x2 = dbg["roi"]
    top, right, bottom, left = dbg["face_box"]
    ratio     = dbg["ratio"]
    threshold = dbg["threshold"]
    edge_h    = dbg["edge_h"]
    edge_v    = dbg["edge_v"]
    margin    = dbg["margin"]
    n_lines   = dbg["n_lines"]
    triggered = dbg["is_screen"]

    inner_density = dbg.get("inner_density", 0.0)
    inner_thresh  = dbg.get("inner_thresh", cfg.SCREEN_INNER_MAX)
    is_border     = dbg.get("is_border", False)
    is_inner      = dbg.get("is_inner", False)

    # ─── วาดเส้นที่ตรวจเจอทั้งหมด ───
    # เทา = edge zone แต่เฉียง (ไม่นับ)
    for fx1, fy1, fx2, fy2 in dbg.get("lines_other", []):
        cv2.line(frame, (fx1, fy1), (fx2, fy2), (100, 100, 100), 1, cv2.LINE_AA)
    # เหลือง = เส้นใน center (นับ inner_density) ← สัญญาณหลักของรูป/วิดีโอ
    for fx1, fy1, fx2, fy2 in dbg.get("lines_inner", []):
        cv2.line(frame, (fx1, fy1), (fx2, fy2), (0, 230, 230), 2, cv2.LINE_AA)
    # เขียว = แนวนอนใน edge zone (นับ edge_H)
    for fx1, fy1, fx2, fy2 in dbg.get("lines_h", []):
        cv2.line(frame, (fx1, fy1), (fx2, fy2), (0, 255, 80), 2, cv2.LINE_AA)
    # ส้ม = แนวตั้งใน edge zone (นับ edge_V)
    for fx1, fy1, fx2, fy2 in dbg.get("lines_v", []):
        cv2.line(frame, (fx1, fy1), (fx2, fy2), (255, 140, 0), 2, cv2.LINE_AA)

    # ─── กรอบ ROI (margin zone) ───
    roi_color = (0, 0, 255) if triggered else (0, 165, 255)
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), roi_color, 1)
    cv2.rectangle(frame, (left, top), (right, bottom), roi_color, 1)

    # ─── ป้าย debug ───
    hud_color = (0, 0, 255) if triggered else (0, 220, 255)

    def _flag(ok): return "FAIL" if ok else "ok"

    fft_score  = dbg.get("fft_score", 1.0)
    fft_thresh = dbg.get("fft_thresh", cfg.FFT_SCORE_MIN)
    peak_rate  = dbg.get("fft_peak_rate", 0.0)
    radial_cov = dbg.get("fft_radial_cov", 0.0)
    is_fft     = dbg.get("is_fft", False)

    texts = [
        f"lines total={n_lines}  margin={margin}px",
        f"[border] H(grn)={edge_h:.0f} V(org)={edge_v:.0f}  ratio={ratio:.4f}/{threshold:.4f} {_flag(is_border)}",
        f"[inner]  cyan={len(dbg.get('lines_inner',[]))}  density={inner_density:.4f}/{inner_thresh:.4f} {_flag(is_inner)}",
        f"[FFT]    score={fft_score:.3f}/{fft_thresh:.3f}  peak={peak_rate:.4f}  rcov={radial_cov:.3f} {_flag(is_fft)}",
        ">>> SCREEN/PHOTO DETECTED! <<<" if triggered else "LIVE  (no screen detected)",
    ]
    x0 = max(roi_x1, 4)
    y0 = roi_y2 + 14
    fh_frame = frame.shape[0]
    if y0 + len(texts) * 17 > fh_frame:
        y0 = max(roi_y1 - len(texts) * 17 - 4, 14)
    for i, txt in enumerate(texts):
        if i == len(texts) - 1:   # บรรทัดสุดท้าย = verdict
            col = (0, 0, 255) if triggered else (0, 220, 100)
        elif i == 1:
            col = (0, 0, 255) if is_border else (160, 160, 160)
        elif i == 2:
            col = (0, 0, 255) if is_inner else (0, 200, 200)
        elif i == 3:
            col = (0, 0, 255) if is_fft else (180, 180, 80)
        else:
            col = hud_color
        cv2.putText(frame, txt, (x0, y0 + i * 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)


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

_panel_cache: dict = {"panel": None, "key": None, "tick": 0}

def draw_screen_debug_panel(panel: np.ndarray, dbg: dict):
    """
    วาดค่า Screen Detection ใน side panel (TEST_MODE)
    dbg ควรเป็นค่าที่ smooth แล้วจาก main.py ไม่กระพริบ
    """
    if not dbg or not dbg.get("valid"):
        return

    W  = panel.shape[1]
    ph = panel.shape[0]

    # ─── header ───
    sec_h  = 145
    y_top  = ph - sec_h
    cv2.rectangle(panel, (0, y_top), (W, y_top + 20), (50, 30, 30), cv2.FILLED)
    cv2.putText(panel, "Screen Debug", (6, y_top + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1)

    def _row(label, val_str, y, flagged=False):
        col_label = (130, 130, 130)
        col_val   = (0, 80, 255) if flagged else (220, 220, 220)
        cv2.putText(panel, label,   (6,      y), cv2.FONT_HERSHEY_SIMPLEX, 0.37, col_label, 1)
        cv2.putText(panel, val_str, (W - 95, y), cv2.FONT_HERSHEY_SIMPLEX, 0.37, col_val,   1)

    y = y_top + 32
    step = 18

    ratio        = dbg.get("ratio", 0.0)
    threshold    = dbg.get("threshold", 0.40)
    inner_d      = dbg.get("inner_density", 0.0)
    inner_t      = dbg.get("inner_thresh", 0.011)
    fft_score    = dbg.get("fft_score", 1.0)
    fft_t        = dbg.get("fft_thresh", 0.50)
    peak_rate    = dbg.get("fft_peak_rate", 0.0)
    radial_cov   = dbg.get("fft_radial_cov", 0.0)
    is_border    = dbg.get("is_border", False)
    is_inner     = dbg.get("is_inner", False)
    is_fft       = dbg.get("is_fft", False)
    triggered    = dbg.get("is_screen", False)

    _row("border ratio",    f"{ratio:.4f} / {threshold:.3f}",    y, is_border);  y += step
    _row("inner density",   f"{inner_d:.4f} / {inner_t:.4f}",    y, is_inner);   y += step
    _row("FFT score",       f"{fft_score:.3f} / {fft_t:.2f}",    y, is_fft);     y += step
    _row("peak_rate",       f"{peak_rate:.4f}",                   y);             y += step
    _row("radial_cov",      f"{radial_cov:.3f}",                  y);             y += step

    # timers
    scr_t   = dbg.get("screen_timer", 0.0)
    real_t  = dbg.get("real_timer",   0.0)
    scr_max = dbg.get("screen_confirm", cfg.SCREEN_CONFIRM_SEC)
    rel_max = dbg.get("real_reset",     cfg.SCREEN_RESET_SEC)
    y += 4
    cv2.line(panel, (4, y), (W - 4, y), (60, 60, 60), 1);  y += 6

    # screen timer bar
    cv2.putText(panel, f"screen {scr_t:.1f}/{scr_max:.0f}s", (6, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                (0, 80, 255) if scr_t > 0 else (80, 80, 80), 1)
    bar_w = int((W - 8) * min(scr_t / max(scr_max, 0.1), 1.0))
    cv2.rectangle(panel, (4, y + 3), (4 + bar_w, y + 9),
                  (0, 60, 200) if scr_t > 0 else (40, 40, 40), cv2.FILLED)
    y += 14

    # real-face reset timer bar
    cv2.putText(panel, f"real   {real_t:.1f}/{rel_max:.1f}s", (6, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                (0, 200, 80) if real_t > 0 else (80, 80, 80), 1)
    bar_w = int((W - 8) * min(real_t / max(rel_max, 0.1), 1.0))
    cv2.rectangle(panel, (4, y + 3), (4 + bar_w, y + 9),
                  (0, 160, 50) if real_t > 0 else (40, 40, 40), cv2.FILLED)
    y += 16

    # verdict bar
    bar_col = (0, 0, 200) if triggered else (0, 160, 60)
    cv2.rectangle(panel, (4, y + 2), (W - 4, y + 18), bar_col, cv2.FILLED)
    verdict = "SCREEN / PHOTO" if triggered else "LIVE  (no spoof)"
    cv2.putText(panel, verdict, (8, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)


def build_panel(persons: dict, liveness_map: dict, frame_height: int,
                screen_debug: dict = None) -> np.ndarray:
    # cache key: rebuild เมื่อ checked_in/out เปลี่ยน หรือ snapshot เปลี่ยน หรือทุก 15 frames
    _panel_cache["tick"] = (_panel_cache["tick"] + 1) % 15
    cache_key = (
        tuple((n, p.checked_in, p.checked_out,
               id(p.snapshot),
               p.last_seen.strftime("%H:%M:%S") if p.last_seen else "-")
              for n, p in persons.items()),
        frame_height,
    )
    # TEST_MODE: ไม่ cache เพื่อให้ screen debug อัปเดตได้ทุก frame
    if not cfg.TEST_MODE:
        if _panel_cache["tick"] != 0 and _panel_cache["panel"] is not None and _panel_cache["key"] == cache_key:
            return _panel_cache["panel"]
    _panel_cache["key"] = cache_key

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

        tx  = xf + FS + 8
        row = y + 14
        GAP = 28 #
        display = getattr(person, "display_name", name) or name
        pid      = str(getattr(person, "employee_code", None) or "-")
        dept     = str(getattr(person, "department",    None) or "-")
        in_str   = person.first_seen.strftime("%H:%M:%S") if person.first_seen else "-"
        lst_str  = person.last_seen.strftime("%H:%M:%S")  if person.last_seen  else "-"

        # Name (Thai) — render บน name ROI เพื่อหลีกเลี่ยงการแปลง full panel เป็น PIL
        name_h = GAP + 4
        name_roi = panel[row-2:row-2+name_h, tx:tx+140]
        if name_roi.size > 0:
            draw_text(name_roi, display[:24], (0, 2), scale=0.36, color=C.TEXT_CYAN)
            panel[row-2:row-2+name_h, tx:tx+140] = name_roi
        row += GAP
        cv2.putText(panel, f"PID: {pid[:15]}", (tx, row),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)
        row += GAP
        cv2.putText(panel, f"Dept:{dept[:13]}", (tx, row),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)
        row += GAP
        cv2.putText(panel, f"IN:  {in_str}", (tx, row),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, C.CHECKED_IN, 1)
        row += GAP
        cv2.putText(panel, f"LST: {lst_str}", (tx, row),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, C.TEXT_DIM, 1)

        y += ITEM_H
        cv2.line(panel, (5, y-4), (W-5, y-4), C.DIVIDER, 1)

    # TEST_MODE: วาด screen debug section ที่ท้าย panel
    if cfg.TEST_MODE and screen_debug:
        draw_screen_debug_panel(panel, screen_debug)

    _panel_cache["panel"] = panel
    return panel