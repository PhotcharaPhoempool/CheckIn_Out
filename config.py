"""
config.py — ตั้งค่าทั้งหมดของระบบ
=============================================
แก้ค่าในไฟล์นี้ไฟล์เดียว ไม่ต้องแก้ไฟล์อื่น
"""
from datetime import time as dtime


# ─── ระบบหลัก ──────────────────────────────
ENCODINGS_FILE        = "encodings.pkl"
FACE_TOLERANCE        = 0.45      # ค่าความเข้มในการเทียบหน้า (ต่ำ=เข้มงวด)
FRAME_SCALE           = 0.25      # ย่อภาพกี่เท่าก่อนตรวจจับ (เล็ก=เร็ว)
PANEL_WIDTH           = 300       # ความกว้าง panel ขวา (px)

# ─── โหมดทดสอบ ─────────────────────────────
TEST_MODE             = False
TEST_DURATION_SECONDS = 60        # OUT หลังกี่วินาที (เฉพาะ TEST_MODE)
CHECKOUT_TIME         = dtime(22, 0)  # เวลา OUT จริง (22:00 = 4 ทุ่ม)

# ─── Performance ───────────────────────────
DETECT_EVERY_N_FRAMES = 2         # ตรวจจับทุกกี่เฟรม (1=ทุกเฟรม, 2=ข้าม 1)
FULLSCREEN            = True      # เปิดเต็มจอ

# ─── Anti-Spoofing: ด่าน 1 — Landmark Depth ───
LIVENESS_TIMEOUT      = 10        # timeout รวม (วินาที)
LIVENESS_RETRY_AFTER  = 4         # รอกี่วินาทีก่อน retry
DEPTH_FRAMES_REQUIRED = 5         # ต้องผ่านกี่เฟรม
DEPTH_FRAMES_WINDOW   = 8         # ดูจากกี่เฟรมล่าสุด
DEPTH_NOSE_MIN        = 0.18      # จมูกยื่นขั้นต่ำ
DEPTH_JAW_MIN         = 0.015     # ความโค้งกรามขั้นต่ำ
DEPTH_SYMMETRY_MIN    = 0.002     # ความไม่สมมาตรขั้นต่ำ

# ─── Anti-Spoofing: ด่าน 2 — Micro-Motion ─────
MOTION_VAR_MIN        = 0.3       # landmark drift ขั้นต่ำ

# ─── Anti-Spoofing: ด่าน 3 — Texture ──────────
TEXTURE_ENABLED       = True
TEXTURE_LBP_MIN       = 3.5       # LBP variance ขั้นต่ำ
TEXTURE_LAP_MIN       = 15.0      # Laplacian variance ขั้นต่ำ
TEXTURE_CHROMA_MIN    = 6.0       # Chroma std ขั้นต่ำ

# ─── Anti-Spoofing: ด่าน 4 — Screen Border ────
SCREEN_DETECT_ENABLED = True
SCREEN_EDGE_MAX       = 0.35      # สัดส่วนเส้นตรงรอบหน้า (เกิน=จอ)

# ─── Anti-Spoofing: ด่าน 5 — Finger Challenge ─
CHALLENGE_ENABLED     = True
CHALLENGE_COUNT       = 1         # ต้องชูนิ้วถูกกี่ครั้ง
CHALLENGE_TIMEOUT     = 6.0       # วินาทีต่อครั้ง
CHALLENGE_HOLD_FRAMES = 4         # ค้างท่าถูกกี่เฟรมถึงนับผ่าน
CHALLENGE_NEAR_FACE   = True      # มือต้องอยู่ใกล้หน้า
CHALLENGE_PROXIMITY   = 1.5       # ระยะมือ (เท่าของความกว้างหน้า)


# ─── สี (BGR) ──────────────────────────────
def _hex(h):
    """แปลง hex เป็น BGR tuple"""
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))

class Color:
    """สีทั้งหมด เรียกใช้: Color.CHECKED_IN"""
    UNKNOWN       = _hex("#FF3333")
    LIVENESS      = _hex("#FFD700")
    LIVENESS_FAIL = _hex("#FF6600")
    CHECKED_IN    = _hex("#00DC00")
    CHECKED_OUT   = _hex("#FF8C00")
    NOT_IN_DB     = _hex("#00C8C8")
    CHALLENGE     = _hex("#FF00FF")
    OK            = _hex("#00FF99")
    FAIL          = _hex("#FF4444")
    # Panel / HUD
    PANEL_BG      = _hex("#1C1C1C")
    PANEL_HEADER  = _hex("#323232")
    DIVIDER       = _hex("#3A3A3A")
    TEXT          = _hex("#FFFFFF")
    TEXT_DIM      = _hex("#BEBEBE")
    TEXT_CYAN     = _hex("#00E6E6")
    TEXT_MORE     = _hex("#969696")
    HUD_TEST      = _hex("#00E6E6")
    HUD_DONE      = _hex("#FF8C00")
    FACE_PH       = _hex("#3C3C3C")