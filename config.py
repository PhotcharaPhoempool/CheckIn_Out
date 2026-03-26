"""
config.py — ตั้งค่าทั้งหมดของระบบ
=============================================
แก้ค่าในไฟล์นี้ไฟล์เดียว ไม่ต้องแก้ไฟล์อื่น
"""
from datetime import time as dtime


# ╔═══════════════════════════════════════════╗
# ║  ระบบหลัก + ArcFace                        ║
# ╚═══════════════════════════════════════════╝
ENCODINGS_FILE        = "encodings.pkl"
FACE_TOLERANCE        = 0.35      # cosine similarity ขั้นต่ำ (ArcFace ใช้ 0.3-0.4)
DET_SIZE              = (320, 320)  # ขนาดภาพสำหรับ detection (320=เร็ว, 640=แม่น)
PANEL_WIDTH           = 250

# ╔═══════════════════════════════════════════╗
# ║  โหมดทดสอบ                                 ║
# ╚═══════════════════════════════════════════╝
TEST_MODE             = False
TEST_DURATION_SECONDS = 60
CHECKOUT_TIME         = dtime(22, 0)

# ╔═══════════════════════════════════════════╗
# ║  Performance                              ║
# ╚═══════════════════════════════════════════╝
DETECT_EVERY_N_FRAMES = 4 #MAX Skip frames between detections
FULLSCREEN            = True
CAMERA_FLIP           = True

# ╔═══════════════════════════════════════════╗
# ║  Anti-Spoofing: ด่าน 1 — Landmark Depth   ║
# ╚═══════════════════════════════════════════╝
LIVENESS_TIMEOUT      = 20
LIVENESS_RETRY_AFTER  = 4
DEPTH_FRAMES_REQUIRED = 5
DEPTH_FRAMES_WINDOW   = 8
DEPTH_NOSE_MIN        = 0.18
DEPTH_JAW_MIN         = 0.015
DEPTH_SYMMETRY_MIN    = 0.002

# ╔═══════════════════════════════════════════╗
# ║  Anti-Spoofing: ด่าน 2 — Micro-Motion     ║
# ╚═══════════════════════════════════════════╝
MOTION_VAR_MIN        = 1.2       # ปรับเพราะ ArcFace ใช้ full-res landmarks (เดิม 0.3 ที่ 0.25x)

# ╔═══════════════════════════════════════════╗
# ║  Anti-Spoofing: ด่าน 3 — Texture          ║
# ╚═══════════════════════════════════════════╝
TEXTURE_ENABLED       = True
TEXTURE_LBP_MIN       = 3.5
TEXTURE_LAP_MIN       = 15.0
TEXTURE_CHROMA_MIN    = 6.0

# ╔═══════════════════════════════════════════╗
# ║  Anti-Spoofing: ด่าน 4 — Screen Border    ║
# ╚═══════════════════════════════════════════╝
SCREEN_DETECT_ENABLED = True
SCREEN_EDGE_MAX       = 0.18

# ╔═══════════════════════════════════════════╗
# ║  Anti-Spoofing: ด่าน 5 — Finger Challenge ║
# ╚═══════════════════════════════════════════╝
CHALLENGE_ENABLED     = True
CHALLENGE_COUNT       = 1
CHALLENGE_TIMEOUT     = 6.0
CHALLENGE_HOLD_FRAMES = 2
CHALLENGE_NEAR_FACE   = True
CHALLENGE_PROXIMITY   = 1.5

# ╔═══════════════════════════════════════════╗
# ║  Anti-Spoofing: ด่าน 6 — MiniFASNet       ║
# ╚═══════════════════════════════════════════╝
FAS_ENABLED           = True
FAS_THRESHOLD         = 0.5
FAS_CHECK_EVERY       = 10
FAS_REQUIRED_REAL     = 2
FAS_DETECTOR_BACKEND  = "skip"

# ╔═══════════════════════════════════════════╗
# ║  UI: General                              ║
# ╚═══════════════════════════════════════════╝
SHOW_LANDMARKS        = False
NO_FACE_RESET_SEC     = 5

# ╔═══════════════════════════════════════════╗
# ║  UI: Face Guide Overlay (วงรี)             ║
# ╚═══════════════════════════════════════════╝
GUIDE_OVERLAY         = True
GUIDE_OVAL_CY         = 0.47     # ตำแหน่งแนวตั้ง (สัดส่วนของ frame height)
GUIDE_OVAL_EW         = 0.29     # ความกว้างวงรี (สัดส่วนของ frame height)
GUIDE_OVAL_EH         = 0.34     # ความสูงวงรี
GUIDE_DIM_FACTOR      = 0.38     # ความมืดนอกวงรี (0=ดำ, 1=ไม่มืด)
GUIDE_DIM_BLUR        = 41       # blur ขอบวงรี (เลขคี่)
GUIDE_IN_OVAL_TOL     = 1.4      # tolerance (1.0=ตรง, 1.5=หลวม)
GUIDE_OVAL_THICK      = 3
GUIDE_OVAL_INNER      = 1

# ╔═══════════════════════════════════════════╗
# ║  UI: Instruction Card (การ์ดด้านล่าง)       ║
# ╚═══════════════════════════════════════════╝
CARD_HEIGHT           = 82
CARD_HEIGHT_SMALL     = 58
CARD_WIDTH_MAX        = 420
CARD_MARGIN_BOTTOM    = 14
CARD_PADDING_LEFT     = 18
CARD_BG               = (245, 245, 245)
CARD_ACCENT_W         = 6
CARD_SHADOW_OFF       = 3
CARD_SHADOW_ALPHA     = 0.35
CARD_FONT_MAIN        = 0.72
CARD_FONT_SUB         = 0.50
CARD_TEXT_MAIN        = (25, 25, 25)
CARD_TEXT_SUB         = (80, 80, 80)
CARD_MAIN_Y           = 14
CARD_SUB_Y            = 50

# ╔═══════════════════════════════════════════╗
# ║  UI: Face Box (กรอบหน้า)                   ║
# ╚═══════════════════════════════════════════╝
FACEBOX_THICK         = 2
FACEBOX_LABEL_H       = 28
FACEBOX_FONT          = 0.45
FACEBOX_PAD           = 5

# ╔═══════════════════════════════════════════╗
# ║  UI: Side Panel (แผงขวา)                   ║
# ╚═══════════════════════════════════════════╝
PANEL_HEADER_H        = 38
PANEL_FACE_SIZE       = 80
PANEL_ITEM_EXTRA      = 58
PANEL_START_Y         = 46
PANEL_FACE_X          = 8

# ╔═══════════════════════════════════════════╗
# ║  สี (BGR)                                 ║
# ╚═══════════════════════════════════════════╝
def _hex(h):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))

class Color:
    UNKNOWN       = _hex("#FF3333")
    LIVENESS      = _hex("#FFD700")
    LIVENESS_FAIL = _hex("#FF6600")
    CHECKED_IN    = _hex("#00DC00")
    CHECKED_OUT   = _hex("#FF8C00")
    NOT_IN_DB     = _hex("#00C8C8")
    CHALLENGE     = _hex("#FF00FF")
    OK            = _hex("#00FF99")
    FAIL          = _hex("#FF4444")
    PANEL_BG      = _hex("#1C1C1C")
    PANEL_HEADER  = _hex("#323232")
    DIVIDER       = _hex("#3A3A3A")
    TEXT          = _hex("#FFFFFF")
    TEXT_DIM      = _hex("#FF0000")
    TEXT_CYAN     = _hex("#00E6E6")
    TEXT_MORE     = _hex("#969696")
    HUD_TEST      = _hex("#00E6E6")
    HUD_DONE      = _hex("#FF8C00")
    FACE_PH       = _hex("#3C3C3C")
    OVAL_DEFAULT  = (210, 210, 210)