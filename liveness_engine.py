"""
liveness_engine.py — ระบบตรวจสอบความจริงของใบหน้า (6 ด่าน)
=============================================================
ด่าน 1: DepthAnalyzer     — ตรวจ 3D geometry จาก landmark
ด่าน 2: MotionAnalyzer    — ตรวจ micro-movement ข้ามเฟรม
ด่าน 3: TextureAnalyzer   — ตรวจ texture ผิวหนัง vs จอ
ด่าน 4: ScreenDetector    — ตรวจขอบจอ/กรอบมือถือ
ด่าน 5: FingerChallenge   — สุ่มชูนิ้ว 1-5
ด่าน 6: FASDetector       — MiniFASNet deep learning (DeepFace)
"""

import math
import random
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from dataclasses import dataclass, field

import config as cfg


# ─────────────────────────────────────────────
# ข้อมูลสถานะของแต่ละคน (1 คน = 1 LivenessState)
# ─────────────────────────────────────────────
@dataclass
class LivenessState:
    """สถานะ liveness ของคนหนึ่งคน — สร้างใหม่ทุกครั้งที่เริ่มตรวจ"""
    start_ts: float = 0.0
    confirmed: bool = False
    failed: bool = False
    fail_reason: str = ""

    # ด่าน 1: Depth
    depth_history: list = field(default_factory=list)
    depth_pass_count: int = 0
    depth_ok: bool = False

    # ด่าน 2: Motion
    centers: deque = field(default_factory=lambda: deque(maxlen=cfg.DEPTH_FRAMES_WINDOW))
    motion_ok: bool = False

    # ด่าน 2.5: Blink
    blink_ok: bool = not cfg.BLINK_ENABLED
    blink_count: int = 0
    _blink_eye_closed: bool = False   # ตาหลับในเฟรมก่อนหน้า

    # ด่าน 3: Texture
    texture_pass: int = 0
    texture_ok: bool = not cfg.TEXTURE_ENABLED

    # ด่าน 4: Screen
    screen_ok: bool = True   # assume OK จนกว่าจะ detect ขอบจอ

    # ด่าน 5: Challenge
    challenge_ok: bool = not cfg.CHALLENGE_ENABLED
    challenge_phase: str = "passive"   # "passive" → "active"
    challenge_number: int = 0          # เลขที่ต้องชู
    challenge_start_ts: float = 0.0
    challenge_done: list = field(default_factory=list)
    challenge_hold: int = 0
    challenge_detected: int = -1       # นิ้วที่ detect ได้ล่าสุด

    # ด่าน 6: MiniFASNet (DeepFace)
    fas_ok: bool = not cfg.FAS_ENABLED
    fas_score: float = -1.0            # score ล่าสุด (-1 = ยังไม่ตรวจ)
    fas_real_count: int = 0            # จำนวนครั้งที่ผ่าน
    fas_check_count: int = 0           # จำนวนครั้งที่ตรวจ

    def is_confirmed(self) -> bool:
        return self.confirmed

    def is_failed(self) -> bool:
        return self.failed

    def all_passive_passed(self) -> bool:
        """ด่าน 1-4 + ด่าน 2.5 + ด่าน 6 ผ่านหมดแล้วหรือยัง"""
        return (self.depth_ok and self.motion_ok and self.blink_ok
                and self.texture_ok and self.screen_ok and self.fas_ok)

    def all_passed(self) -> bool:
        """ทุกด่านผ่าน"""
        return self.all_passive_passed() and self.challenge_ok


# ─────────────────────────────────────────────
# ด่าน 1: Landmark Depth
# ─────────────────────────────────────────────
class DepthAnalyzer:
    """ตรวจสัดส่วน 3D ของใบหน้าจาก 2D landmark"""

    @staticmethod
    def _dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _point_line_dist(point, line_start, line_end):
        """ระยะจากจุดถึงเส้นตรง"""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 1:
            return 0.0
        return abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / length

    def nose_protrusion(self, lm: dict) -> float:
        """จมูกยื่นจากระนาบหน้า — หน้าจริง > 0.18, รูปแบน < 0.18"""
        nose = lm.get("nose_bridge", [])
        chin = lm.get("chin", [])
        if len(nose) < 2 or len(chin) < 17:
            return 0.0
        face_w = self._dist(chin[0], chin[16])
        if face_w < 1:
            return 0.0
        dist = self._point_line_dist(nose[-1], chin[0], chin[16])
        return dist / face_w

    def jaw_curvature(self, lm: dict) -> float:
        """ความโค้งของกราม — หน้าจริงโค้ง 3D"""
        chin = lm.get("chin", [])
        if len(chin) < 17:
            return 0.0
        total = 0.0
        for side in [chin[0:9], chin[8:17]]:
            seg_len = self._dist(side[0], side[-1])
            if seg_len < 1:
                continue
            max_dev = max(
                self._point_line_dist(p, side[0], side[-1])
                for p in side[1:-1]
            )
            total += max_dev / seg_len
        return total / 2.0

    def symmetry_variance(self, lm: dict) -> float:
        """ความไม่สมมาตร — หน้าจริงไม่สมมาตรเล็กน้อย"""
        chin = lm.get("chin", [])
        l_eye = lm.get("left_eye", [])
        r_eye = lm.get("right_eye", [])
        if len(chin) < 17 or len(l_eye) < 6 or len(r_eye) < 6:
            return 0.0

        mid_x = (chin[0][0] + chin[16][0]) / 2.0
        ratios = []

        # ตา
        lc = sum(p[0] for p in l_eye) / len(l_eye)
        rc = sum(p[0] for p in r_eye) / len(r_eye)
        dl, dr = abs(lc - mid_x), abs(rc - mid_x)
        if dl + dr > 0:
            ratios.append(abs(dl - dr) / (dl + dr))

        # กราม
        for i in range(1, 8):
            dl = abs(chin[i][0] - mid_x)
            dr = abs(chin[16-i][0] - mid_x)
            if dl + dr > 0:
                ratios.append(abs(dl - dr) / (dl + dr))

        return float(np.var(ratios)) if ratios else 0.0

    def analyze(self, lm: dict) -> dict:
        """ตรวจทั้ง 3 ค่า — ผ่าน 2/3 ถือว่า OK"""
        nose = self.nose_protrusion(lm)
        jaw  = self.jaw_curvature(lm)
        sym  = self.symmetry_variance(lm)
        checks = [
            nose >= cfg.DEPTH_NOSE_MIN,
            jaw  >= cfg.DEPTH_JAW_MIN,
            sym  >= cfg.DEPTH_SYMMETRY_MIN,
        ]
        return {
            "nose": nose, "jaw": jaw, "sym": sym,
            "is_3d": sum(checks) >= 2,
        }


# ─────────────────────────────────────────────
# ด่าน 2: Micro-Motion
# ─────────────────────────────────────────────
class MotionAnalyzer:
    """ตรวจว่าหน้าขยับจริง (ไม่ใช่รูปนิ่ง)"""

    @staticmethod
    def landmark_center(lm: dict) -> tuple:
        pts = [p for group in lm.values() for p in group]
        if not pts:
            return (0.0, 0.0)
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        return (cx, cy)

    @staticmethod
    def compute_motion(centers: deque) -> float:
        """คำนวณ motion variance จาก landmark centers หลายเฟรม"""
        if len(centers) < 4:
            return 0.0
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        return float(np.std(xs) + np.std(ys))


# ─────────────────────────────────────────────
# ด่าน 2.5: Blink Detection (Eye Aspect Ratio)
# ─────────────────────────────────────────────
class BlinkDetector:
    """
    ตรวจการกะพริบตา ด้วย Eye Aspect Ratio (EAR)
    ใช้ left_eye + right_eye จาก 68-point landmark ที่มีอยู่แล้ว — ไม่กิน CPU เพิ่ม

    EAR = (||p1-p5|| + ||p2-p4||) / (2 * ||p0-p3||)
    ตาเปิด: EAR ≈ 0.28-0.35   ตาหลับ: EAR < BLINK_EAR_THRESH

    รูปภาพ/วิดีโอนิ่ง: EAR คงที่ ไม่มี blink → ไม่ผ่าน
    """

    @staticmethod
    def _dist(a, b) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @classmethod
    def ear(cls, eye_pts: list) -> float:
        """คำนวณ Eye Aspect Ratio จาก 6 จุด (dlib/InsightFace order)"""
        if len(eye_pts) < 6:
            return 1.0   # assume open ถ้าไม่มีข้อมูล
        p = eye_pts
        vertical = cls._dist(p[1], p[5]) + cls._dist(p[2], p[4])
        horizontal = cls._dist(p[0], p[3])
        return vertical / (2.0 * horizontal + 1e-6)

    @classmethod
    def avg_ear(cls, lm: dict) -> float:
        """EAR เฉลี่ยตาซ้าย-ขวา"""
        l = cls.ear(lm.get("left_eye", []))
        r = cls.ear(lm.get("right_eye", []))
        return (l + r) / 2.0


# ─────────────────────────────────────────────
# ด่าน 3: Texture Analysis
# ─────────────────────────────────────────────
class TextureAnalyzer:
    """ตรวจ micro-texture ผิวหนัง — จอมือถือมี pixel pattern ต่างจากหน้าจริง"""

    FACE_SIZE = 64   # resize ให้คงที่ก่อนวิเคราะห์ (64 เพียงพอ ลด compute ~44%)

    def _lbp_variance(self, gray: np.ndarray) -> float:
        """Local Binary Pattern variance"""
        f = cv2.resize(gray, (self.FACE_SIZE, self.FACE_SIZE)).astype(np.int16)
        lbp = np.zeros_like(f, dtype=np.uint8)
        for i, (dy, dx) in enumerate([
            (-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)
        ]):
            shifted = np.roll(np.roll(f, dy, axis=0), dx, axis=1)
            lbp += ((shifted >= f).astype(np.uint8)) << i
        return float(np.std(lbp[1:-1, 1:-1].astype(np.float64)))

    def _laplacian_variance(self, gray: np.ndarray) -> float:
        """High-frequency energy"""
        face = cv2.resize(gray, (self.FACE_SIZE, self.FACE_SIZE))
        return float(np.var(cv2.Laplacian(face, cv2.CV_64F)))

    def _chroma_std(self, bgr: np.ndarray) -> float:
        """Color chrominance variation"""
        face = cv2.resize(bgr, (self.FACE_SIZE, self.FACE_SIZE))
        ycrcb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
        cr = float(np.std(ycrcb[:, :, 1]))
        cb = float(np.std(ycrcb[:, :, 2]))
        return (cr + cb) / 2.0

    def analyze(self, bgr_face: np.ndarray) -> bool:
        """ผ่าน 2/3 เกณฑ์ = หน้าจริง"""
        if bgr_face.shape[0] < 10 or bgr_face.shape[1] < 10:
            return False
        gray = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2GRAY)
        checks = [
            self._lbp_variance(gray)     >= cfg.TEXTURE_LBP_MIN,
            self._laplacian_variance(gray) >= cfg.TEXTURE_LAP_MIN,
            self._chroma_std(bgr_face)    >= cfg.TEXTURE_CHROMA_MIN,
        ]
        return sum(checks) >= 2


# ─────────────────────────────────────────────
# ด่าน 4: Screen Border Detection
# ─────────────────────────────────────────────
class ScreenDetector:
    """ตรวจขอบจอ/กรอบมือถือรอบใบหน้า ด้วย Canny + HoughLines"""

    def detect(self, bgr_frame: np.ndarray, face_box: tuple) -> tuple:
        """
        Returns: (is_screen: bool, ratio: float)
        face_box = (top, right, bottom, left)

        ตรวจเฉพาะเส้นที่อยู่ในแถบขอบ ROI (margin zone) รอบหน้า:
          - Phone/tablet border จะอยู่ชิดหน้า → ตกใน margin zone → นับ
          - Background (ชั้น, จอคอม, ผนัง) อยู่ไกลกว่า → ตกนอก margin zone → ไม่นับ
        """
        margin = cfg.SCREEN_MARGIN
        top, right, bottom, left = face_box
        fh, fw = bgr_frame.shape[:2]

        y1 = max(0, top - margin)
        y2 = min(fh, bottom + margin)
        x1 = max(0, left - margin)
        x2 = min(fw, right + margin)

        roi_bgr = bgr_frame[y1:y2, x1:x2]
        if roi_bgr.size == 0:
            return False, 0.0
        rh, rw = roi_bgr.shape[:2]
        roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(roi, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=30,
            minLineLength=min(rh, rw) // 4,
            maxLineGap=8,
        )
        if lines is None:
            return False, 0.0

        # แบ่ง ROI เป็น "edge zone" (margin px) รอบนอก กับ "center" (พื้นที่หน้า)
        # นับเฉพาะเส้นที่ midpoint อยู่ใน edge zone — phone border อยู่ที่นี่
        edge_h = 0.0   # เส้นแนวนอนในแถบขอบ
        edge_v = 0.0   # เส้นแนวตั้งในแถบขอบ
        for line in lines:
            x1l, y1l, x2l, y2l = line[0]
            mx, my = (x1l + x2l) / 2, (y1l + y2l) / 2
            length = math.hypot(x2l - x1l, y2l - y1l)
            angle  = abs(math.atan2(y2l - y1l, x2l - x1l))

            # เส้นอยู่ใน edge zone หรือไม่
            in_h_zone = my < margin or my > rh - margin   # แถบบน/ล่าง
            in_v_zone = mx < margin or mx > rw - margin   # แถบซ้าย/ขวา

            if angle < 0.18 or abs(angle - math.pi) < 0.18:   # แนวนอน
                if in_h_zone:
                    edge_h += length
            elif abs(angle - math.pi / 2) < 0.18:             # แนวตั้ง
                if in_v_zone:
                    edge_v += length

        perimeter = 2 * (rh + rw)
        ratio = min(edge_h, edge_v) / perimeter if perimeter > 0 else 0
        return ratio > cfg.SCREEN_EDGE_MAX, ratio


# ─────────────────────────────────────────────
# ด่าน 6: MiniFASNet via DeepFace
# ─────────────────────────────────────────────
class FASDetector:
    """
    ตรวจ face anti-spoofing ด้วย MiniFASNet (ผ่าน DeepFace)
    
    Model ขนาด ~1.5 MB, รันบน CPU ได้
    ให้ score 0.0-1.0 (> 0.5 = REAL, < 0.5 = SPOOF)
    
    ใช้งาน:
        fas = FASDetector()
        result = fas.check(face_crop_bgr)
        # result = {"is_real": True, "score": 0.85} หรือ None
    """

    def __init__(self):
        self._deepface = None
        self._loaded = False
        self._available = True

    @property
    def is_available(self) -> bool:
        """FAS พร้อมใช้งานไหม (lazy load ครั้งแรก)"""
        if not self._loaded:
            self._ensure_loaded()
        return self._available

    def _ensure_loaded(self):
        """Lazy load DeepFace (ครั้งแรกจะ download model อัตโนมัติ)"""
        if self._loaded:
            return self._available
        self._loaded = True

        try:
            from deepface import DeepFace
            self._deepface = DeepFace

            # Warm up — ให้ download model ก่อน
            dummy = np.zeros((80, 80, 3), dtype=np.uint8)
            dummy[20:60, 20:60] = 128
            self._deepface.extract_faces(
                dummy, anti_spoofing=True,
                enforce_detection=False,
                detector_backend="skip",
            )
            print("[FAS] MiniFASNet โหลดสำเร็จ")
            return True

        except ImportError:
            print("[FAS] WARNING: ไม่พบ deepface — pip install deepface tf-keras")
            print("[FAS] ด่าน 6 จะถูกข้ามไป")
            self._available = False
            return False

        except Exception as e:
            print(f"[FAS] WARNING: โหลด model ไม่สำเร็จ: {e}")
            self._available = False
            return False

    def check(self, face_crop_bgr: np.ndarray) -> dict:
        """
        ตรวจ face crop ว่าเป็นของจริงหรือไม่
        
        Args:
            face_crop_bgr: ภาพหน้าที่ตัดมาแล้ว (BGR, ขนาดใดก็ได้)
        
        Returns:
            {"is_real": bool, "score": float} หรือ None ถ้าตรวจไม่ได้
        """
        if not self._ensure_loaded():
            return None

        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return None
        if face_crop_bgr.shape[0] < 30 or face_crop_bgr.shape[1] < 30:
            return None

        try:
            # แปลง BGR → RGB (DeepFace ใช้ RGB)
            rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)

            results = self._deepface.extract_faces(
                img_path=rgb,
                anti_spoofing=True,
                enforce_detection=False,
                detector_backend=cfg.FAS_DETECTOR_BACKEND,
            )

            if not results:
                return None

            face = results[0]
            score = face.get("antispoof_score", 0.0)
            is_real = face.get("is_real", score > cfg.FAS_THRESHOLD)

            return {"is_real": is_real, "score": score}

        except Exception:
            return None


# ─────────────────────────────────────────────
# ด่าน 5: Finger Challenge (MediaPipe)
# ─────────────────────────────────────────────
class FingerChallenge:
    """สุ่มให้ชูนิ้ว 1-5 ป้องกันวิดีโอ"""

    _mp_hands = mp.solutions.hands

    # landmark indices
    TIPS = [
        _mp_hands.HandLandmark.THUMB_TIP,
        _mp_hands.HandLandmark.INDEX_FINGER_TIP,
        _mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        _mp_hands.HandLandmark.RING_FINGER_TIP,
        _mp_hands.HandLandmark.PINKY_TIP,
    ]
    PIPS = [
        _mp_hands.HandLandmark.THUMB_IP,
        _mp_hands.HandLandmark.INDEX_FINGER_PIP,
        _mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        _mp_hands.HandLandmark.RING_FINGER_PIP,
        _mp_hands.HandLandmark.PINKY_PIP,
    ]

    @classmethod
    def count_fingers(cls, hand_lm, handedness: str = "Right") -> int:
        """นับนิ้วที่ชูขึ้น (0-5)"""
        lm = hand_lm.landmark
        count = 0

        # นิ้วโป้ง (แนวนอน)
        tip = lm[cls.TIPS[0]]
        ip  = lm[cls.PIPS[0]]
        if handedness == "Right":
            count += int(tip.x < ip.x)
        else:
            count += int(tip.x > ip.x)

        # นิ้วชี้–ก้อย (แนวตั้ง)
        for i in range(1, 5):
            if lm[cls.TIPS[i]].y < lm[cls.PIPS[i]].y:
                count += 1

        return count

    @staticmethod
    def hand_center_px(hand_lm, img_w: int, img_h: int) -> tuple:
        """จุดศูนย์กลางมือ (pixel)"""
        pts = hand_lm.landmark
        cx = sum(p.x for p in pts) / len(pts) * img_w
        cy = sum(p.y for p in pts) / len(pts) * img_h
        return (cx, cy)

    @staticmethod
    def is_near_face(hand_center: tuple, face_box: tuple, face_width: float) -> bool:
        """มืออยู่ใกล้หน้าพอไหม"""
        top, right, bottom, left = face_box
        face_cx = (left + right) / 2.0
        face_cy = (top + bottom) / 2.0
        dist = math.hypot(hand_center[0] - face_cx, hand_center[1] - face_cy)
        return dist < face_width * cfg.CHALLENGE_PROXIMITY

    @staticmethod
    def pick_number(exclude: int = 0) -> int:
        """สุ่มเลข 1-5 ที่ไม่ซ้ำกับเลขก่อนหน้า"""
        num = random.randint(1, 5)
        while num == exclude:
            num = random.randint(1, 5)
        return num


# ─────────────────────────────────────────────
# LivenessEngine — ประสาน 5 ด่านเข้าด้วยกัน
# ─────────────────────────────────────────────
class LivenessEngine:
    """
    ใช้งาน:
        engine = LivenessEngine()
        state  = engine.create_state(now_ts)
        engine.update(state, lm, crop, face_box, frame, hand_results, now_ts, do_detect)
    """

    def __init__(self):
        self.depth    = DepthAnalyzer()
        self.motion   = MotionAnalyzer()
        self.blink    = BlinkDetector()
        self.texture  = TextureAnalyzer()
        self.screen   = ScreenDetector()
        self.finger   = FingerChallenge()
        self.fas      = FASDetector()

    def create_state(self, now_ts: float) -> LivenessState:
        return LivenessState(start_ts=now_ts)

    def update(self, s: LivenessState, lm: dict, face_crop: np.ndarray,
               face_box: tuple, face_width: int, frame: np.ndarray,
               hand_results, now_ts: float, do_detect: bool):
        """อัปเดตสถานะ liveness ทุกเฟรม — เรียกจาก main loop"""

        if s.confirmed or s.failed:
            return

        # ─── Timeout ───
        if now_ts - s.start_ts > cfg.LIVENESS_TIMEOUT:
            s.failed = True
            reasons = []
            if not s.depth_ok:     reasons.append("Depth")
            if not s.motion_ok:    reasons.append("Motion")
            if not s.blink_ok:     reasons.append("Blink")
            if not s.texture_ok:   reasons.append("Texture")
            if not s.screen_ok:    reasons.append("Screen")
            if not s.fas_ok:       reasons.append("FAS")
            if not s.challenge_ok: reasons.append("Finger")
            s.fail_reason = "FAIL:" + "+".join(reasons) if reasons else "Timeout"
            print(f"[TIMEOUT] {s.fail_reason}")
            return

        # ─── ด่าน 1: Depth ───
        if not s.depth_ok and do_detect:
            result = self.depth.analyze(lm)
            s.depth_history.append(result["is_3d"])
            if len(s.depth_history) > cfg.DEPTH_FRAMES_WINDOW:
                s.depth_history = s.depth_history[-cfg.DEPTH_FRAMES_WINDOW:]
            s.depth_pass_count = sum(s.depth_history)
            if s.depth_pass_count >= cfg.DEPTH_FRAMES_REQUIRED:
                s.depth_ok = True
                print(f"[DEPTH OK] nose={result['nose']:.3f} jaw={result['jaw']:.4f}")

        # ─── ด่าน 2: Motion ───
        if not s.motion_ok and do_detect:
            s.centers.append(self.motion.landmark_center(lm))
            motion_val = self.motion.compute_motion(s.centers)
            if motion_val >= cfg.MOTION_VAR_MIN:
                s.motion_ok = True
                print(f"[MOTION OK] var={motion_val:.2f}")

        # ─── ด่าน 2.5: Blink ───
        if cfg.BLINK_ENABLED and not s.blink_ok and do_detect and lm:
            ear_val = BlinkDetector.avg_ear(lm)
            if ear_val < cfg.BLINK_EAR_THRESH:
                s._blink_eye_closed = True
            elif s._blink_eye_closed:
                # ตาเพิ่งเปิด → บลิงค์ครั้งหนึ่งสมบูรณ์
                s.blink_count += 1
                s._blink_eye_closed = False
                print(f"[BLINK] {s.blink_count}/{cfg.BLINK_MIN}  EAR={ear_val:.3f}")
                if s.blink_count >= cfg.BLINK_MIN:
                    s.blink_ok = True
                    print("[BLINK OK]")

        # ─── ด่าน 3: Texture (เฉพาะ detection frames — ลด CPU load) ───
        if cfg.TEXTURE_ENABLED and not s.texture_ok and do_detect:
            if face_crop is not None and face_crop.size > 0:
                if self.texture.analyze(face_crop):
                    s.texture_pass += 1
                if s.texture_pass >= 3:
                    s.texture_ok = True
                    print("[TEXTURE OK]")

        # ─── ด่าน 4: Screen Border ───
        if cfg.SCREEN_DETECT_ENABLED and s.screen_ok and do_detect:
            is_screen, ratio = self.screen.detect(frame, face_box)
            if is_screen:
                s.screen_ok = False
                s.failed = True
                s.fail_reason = f"Screen({ratio:.2f})"
                print(f"[SCREEN DETECTED] ratio={ratio:.2f}")
                return

        # ─── ด่าน 6: MiniFASNet (DeepFace) ───
        if cfg.FAS_ENABLED and not s.fas_ok and not s.failed and do_detect:
            # ถ้า DeepFace ไม่พร้อม → ข้ามด่านนี้ ไม่ block ระบบ
            if not self.fas.is_available:
                s.fas_ok = True
            else:
                s.fas_check_count += 1
                if s.fas_check_count % cfg.FAS_CHECK_EVERY == 0:
                    result = self.fas.check(face_crop)
                    if result:
                        s.fas_score = result["score"]
                        if result["is_real"]:
                            s.fas_real_count += 1
                            if s.fas_real_count >= cfg.FAS_REQUIRED_REAL:
                                s.fas_ok = True
                                print(f"[FAS OK] score={result['score']:.3f} ({s.fas_real_count} real)")
                        else:
                            s.failed = True
                            s.fail_reason = f"FAS:Spoof({result['score']:.2f})"
                            print(f"[FAS SPOOF] score={result['score']:.3f}")
                            return

        # ─── ด่าน 5: Finger Challenge ───
        if cfg.CHALLENGE_ENABLED and not s.challenge_ok and s.all_passive_passed() and not s.failed:
            self._update_challenge(s, hand_results, face_box, face_width, frame, now_ts)

        # ─── ตรวจว่าผ่านทุกด่าน ───
        if s.all_passed() and not s.confirmed:
            s.confirmed = True
            print("[LIVENESS OK] — ผ่านทุกด่าน")

    def _update_challenge(self, s: LivenessState, hand_results,
                          face_box: tuple, face_width: int,
                          frame: np.ndarray, now_ts: float):
        """จัดการ finger challenge"""

        # เริ่ม challenge ครั้งแรก
        if s.challenge_phase == "passive":
            s.challenge_phase = "active"
            s.challenge_number = self.finger.pick_number()
            s.challenge_start_ts = now_ts
            s.challenge_hold = 0
            print(f"[CHALLENGE] Show {s.challenge_number} finger(s)")
            return

        if s.challenge_phase != "active":
            return

        # Timeout challenge นี้
        if now_ts - s.challenge_start_ts > cfg.CHALLENGE_TIMEOUT:
            s.failed = True
            s.fail_reason = f"Finger:{s.challenge_number}"
            print(f"[CHALLENGE FAIL] timeout → {s.challenge_number}")
            return

        # ตรวจนิ้ว
        if not hand_results or not hand_results.multi_hand_landmarks:
            return

        for i, hlm in enumerate(hand_results.multi_hand_landmarks):
            # Handedness
            h_label = "Right"
            if hand_results.multi_handedness:
                h_label = hand_results.multi_handedness[i].classification[0].label

            fingers = self.finger.count_fingers(hlm, h_label)
            s.challenge_detected = fingers

            # เช็คว่ามืออยู่ใกล้หน้า
            if cfg.CHALLENGE_NEAR_FACE:
                hc = self.finger.hand_center_px(hlm, frame.shape[1], frame.shape[0])
                if not self.finger.is_near_face(hc, face_box, face_width):
                    continue

            if fingers == s.challenge_number:
                s.challenge_hold += 1
                if s.challenge_hold >= cfg.CHALLENGE_HOLD_FRAMES:
                    s.challenge_done.append(s.challenge_number)
                    print(f"[CHALLENGE PASS] {fingers} fingers ({len(s.challenge_done)}/{cfg.CHALLENGE_COUNT})")

                    if len(s.challenge_done) >= cfg.CHALLENGE_COUNT:
                        s.challenge_ok = True
                        print("[CHALLENGE OK]")
                    else:
                        s.challenge_number = self.finger.pick_number(exclude=s.challenge_number)
                        s.challenge_start_ts = now_ts
                        s.challenge_hold = 0
                        print(f"[CHALLENGE] Show {s.challenge_number} finger(s)")
            else:
                s.challenge_hold = 0
            break  # ใช้แค่มือแรก