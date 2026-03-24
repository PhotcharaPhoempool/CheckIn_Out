"""
test_gaussian_fourier.py — ทดสอบ Gaussian + Fourier แยกหน้าจริง vs หน้าจอ
=============================================================================
ดูผลของ Gaussian Blur ต่อ Fourier spectrum แบบ real-time
เปรียบเทียบ Raw vs Blurred ข้างกัน

วิธีใช้:
  python test_gaussian_fourier.py              ← กล้อง 0
  python test_gaussian_fourier.py --cam 1      ← กล้อง 1

ปุ่มกด:
  q     = ออก
  s     = screenshot
  +/-   = เพิ่ม/ลด Gaussian kernel size
  r     = reset สถิติ

วิธีทดสอบ:
  1. หันหน้าจริงเข้ากล้อง → ดูค่า
  2. เอามือถือเปิดรูป/วิดีโอมาส่อง → ดูค่า
  3. เปรียบเทียบว่า Gaussian ช่วยแยกได้ชัดขึ้นไหม
"""

import cv2
import numpy as np
import argparse
import time


# ─── ตั้งค่า ───────────────────────────────
GAUSSIAN_KERNEL = 5     # ขนาด blur (ต้องเป็นเลขคี่: 3, 5, 7, 9...)
ANALYZE_SIZE    = 128   # resize หน้าก่อนวิเคราะห์


def fourier_analyze(gray_face: np.ndarray) -> dict:
    """วิเคราะห์ Fourier spectrum ของภาพ grayscale"""
    face = cv2.resize(gray_face, (ANALYZE_SIZE, ANALYZE_SIZE))

    # FFT
    f = np.fft.fft2(face.astype(np.float32))
    f_shift = np.fft.fftshift(f)
    mag = np.abs(f_shift)
    mag_log = np.log1p(mag)

    h, w = mag.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 4

    # Low/High freq masks
    Y, X = np.ogrid[:h, :w]
    low_mask  = ((X - cx)**2 + (Y - cy)**2) <= radius**2
    high_mask = ~low_mask

    # 1. High-freq energy ratio
    total_e = np.sum(mag**2) + 1e-10
    high_e  = np.sum(mag[high_mask]**2)
    hi_ratio = high_e / total_e

    # 2. Moiré score
    hv = mag[high_mask]
    moire = np.max(hv) / (np.median(hv) + 1e-10) if len(hv) > 0 else 0

    # 3. Spectral flatness
    flat = mag.flatten()
    flat = flat[flat > 0]
    if len(flat) > 0:
        geo = np.exp(np.mean(np.log(flat + 1e-10)))
        ari = np.mean(flat)
        flatness = geo / ari if ari > 0 else 0
    else:
        flatness = 0

    # 4. Azimuthal average (1D power spectrum)
    max_r = min(cx, cy)
    radial = np.zeros(max_r)
    count  = np.zeros(max_r)
    for iy in range(h):
        for ix in range(w):
            r = int(np.sqrt((ix - cx)**2 + (iy - cy)**2))
            if r < max_r:
                radial[r] += mag[iy, ix]
                count[r]  += 1
    count[count == 0] = 1
    radial /= count

    return {
        "hi_ratio": hi_ratio,
        "moire":    moire,
        "flatness": flatness,
        "mag_log":  mag_log,
        "radial":   radial,
    }


def draw_spectrum(mag_log, size=130):
    """สร้างภาพ spectrum"""
    norm = cv2.normalize(mag_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img = cv2.resize(norm, (size, size))
    return cv2.applyColorMap(img, cv2.COLORMAP_JET)


def draw_radial_plot(radial, size_w=260, size_h=100, color=(0, 255, 0)):
    """วาดกราฟ 1D radial power spectrum"""
    plot = np.zeros((size_h, size_w, 3), dtype=np.uint8)
    plot[:] = (30, 30, 30)

    if len(radial) == 0:
        return plot

    # Normalize
    r = radial.copy()
    r = r / (np.max(r) + 1e-10)

    pts = []
    for i, v in enumerate(r):
        x = int(i / len(r) * (size_w - 10)) + 5
        y = size_h - 5 - int(v * (size_h - 15))
        pts.append((x, y))

    for i in range(len(pts) - 1):
        cv2.line(plot, pts[i], pts[i+1], color, 1)

    cv2.putText(plot, "Low freq", (5, size_h - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (100, 100, 100), 1)
    cv2.putText(plot, "High freq", (size_w - 60, size_h - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (100, 100, 100), 1)

    return plot


def make_analysis_panel(label, face_gray, face_bgr, panel_w=280, panel_h=480):
    """สร้าง panel วิเคราะห์ 1 ชุด (Raw หรือ Blurred)"""
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (25, 25, 25)

    # Header
    cv2.rectangle(panel, (0, 0), (panel_w, 30), (50, 50, 50), -1)
    cv2.putText(panel, label, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Face thumbnail
    thumb = cv2.resize(face_bgr, (80, 80))
    panel[40:120, 10:90] = thumb

    # Analyze
    result = fourier_analyze(face_gray)

    # Spectrum
    spec = draw_spectrum(result["mag_log"], size=80)
    panel[40:120, 100:180] = spec

    # Radial plot
    radial_color = (0, 200, 0) if "Raw" in label else (0, 150, 255)
    radial_plot = draw_radial_plot(result["radial"], size_w=panel_w-20, size_h=80,
                                   color=radial_color)
    panel[130:210, 10:panel_w-10] = radial_plot
    cv2.putText(panel, "Radial Power Spectrum", (10, 127),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

    # ค่าตัวเลข
    y = 230
    vals = [
        ("HiFreq Energy", result["hi_ratio"], f"{result['hi_ratio']:.4f}"),
        ("Moire Score",    result["moire"],    f"{result['moire']:.3f}"),
        ("Spectral Flat",  result["flatness"], f"{result['flatness']:.4f}"),
    ]

    for name, val, txt in vals:
        cv2.putText(panel, f"{name}:", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
        cv2.putText(panel, txt, (160, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        y += 22

    return panel, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"เปิดกล้อง {args.cam} ไม่ได้!")
        return

    global GAUSSIAN_KERNEL

    print("=" * 55)
    print("  Gaussian + Fourier Analysis Test")
    print("=" * 55)
    print(f"  กล้อง: {args.cam}")
    print(f"  Gaussian kernel: {GAUSSIAN_KERNEL}")
    print(f"  [+/-] ปรับ kernel | [q] ออก | [s] screenshot | [r] reset")
    print("=" * 55)

    # Face detection
    try:
        import face_recognition
        use_fr = True
        print("  Face detection: face_recognition")
    except ImportError:
        use_fr = False
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        print("  Face detection: Haar cascade")

    fps_cnt = 0
    fps_t   = time.time()
    fps     = 0

    # สถิติเปรียบเทียบ
    stats = {"raw_hi": [], "blur_hi": [], "raw_moire": [], "blur_moire": [],
             "raw_flat": [], "blur_flat": []}

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        fps_cnt += 1
        now = time.time()
        if now - fps_t >= 1.0:
            fps = fps_cnt / (now - fps_t)
            fps_cnt = 0
            fps_t = now

        # ─── หาหน้า ───
        if use_fr:
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb)
            faces = [(l*4, t*4, (r-l)*4, (b-t)*4) for (t, r, b, l) in locs]
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = cascade.detectMultiScale(gray, 1.3, 5)
            faces = [(x, y, w, h) for (x, y, w, h) in detected]

        display = frame.copy()

        panel_h = frame.shape[0]
        panel_w = 280

        if faces:
            x, y, w, h = faces[0]  # ใช้หน้าแรก
            pad = 10
            fh, fw = frame.shape[:2]
            x1, y1 = max(0, x-pad), max(0, y-pad)
            x2, y2 = min(fw, x+w+pad), min(fh, y+h+pad)
            crop = frame[y1:y2, x1:x2]

            if crop.size > 0 and crop.shape[0] > 30 and crop.shape[1] > 30:
                # Raw
                gray_raw = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

                # Gaussian Blurred
                blurred_bgr  = cv2.GaussianBlur(crop, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0)
                gray_blurred = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2GRAY)

                # สร้าง panel ซ้าย (Raw) + ขวา (Blurred)
                panel_raw, res_raw   = make_analysis_panel("Raw Image", gray_raw, crop,
                                                            panel_w, panel_h)
                panel_blur, res_blur = make_analysis_panel(
                    f"Gaussian k={GAUSSIAN_KERNEL}", gray_blurred, blurred_bgr,
                    panel_w, panel_h)

                # เก็บสถิติ
                stats["raw_hi"].append(res_raw["hi_ratio"])
                stats["blur_hi"].append(res_blur["hi_ratio"])
                stats["raw_moire"].append(res_raw["moire"])
                stats["blur_moire"].append(res_blur["moire"])
                stats["raw_flat"].append(res_raw["flatness"])
                stats["blur_flat"].append(res_blur["flatness"])
                # เก็บแค่ 200 ล่าสุด
                for k in stats:
                    if len(stats[k]) > 200:
                        stats[k] = stats[k][-200:]

                # วาดกรอบหน้า
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 255), 2)

                # ─── Diff panel ─── แสดงความต่าง Raw vs Blurred
                diff_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
                diff_panel[:] = (20, 20, 20)
                cv2.rectangle(diff_panel, (0, 0), (panel_w, 30), (60, 40, 40), -1)
                cv2.putText(diff_panel, "Comparison", (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                dy = 50
                comparisons = [
                    ("HiFreq", res_raw["hi_ratio"], res_blur["hi_ratio"],
                     "lower=better (screen has higher)"),
                    ("Moire", res_raw["moire"], res_blur["moire"],
                     "lower=better (screen has higher)"),
                    ("Flatness", res_raw["flatness"], res_blur["flatness"],
                     "higher=better (screen has lower)"),
                ]

                for name, raw_v, blur_v, hint in comparisons:
                    diff = blur_v - raw_v
                    cv2.putText(diff_panel, f"{name}:", (10, dy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

                    # Raw value
                    cv2.putText(diff_panel, f"Raw:  {raw_v:.4f}", (10, dy+18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 0), 1)
                    # Blur value
                    cv2.putText(diff_panel, f"Blur: {blur_v:.4f}", (10, dy+36),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 150, 255), 1)
                    # Diff
                    diff_color = (0, 255, 255)
                    cv2.putText(diff_panel, f"Diff: {diff:+.4f}", (150, dy+18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, diff_color, 1)
                    # Hint
                    cv2.putText(diff_panel, hint, (10, dy+52),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (100, 100, 100), 1)

                    dy += 72

                # ─── Average stats ───
                if len(stats["raw_hi"]) > 5:
                    dy += 10
                    cv2.putText(diff_panel, f"Avg ({len(stats['raw_hi'])} frames):",
                                (10, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
                    dy += 20
                    for name, rk, bk in [("HiFreq","raw_hi","blur_hi"),
                                          ("Moire","raw_moire","blur_moire"),
                                          ("Flat","raw_flat","blur_flat")]:
                        rv = np.mean(stats[rk])
                        bv = np.mean(stats[bk])
                        cv2.putText(diff_panel, f"  {name}  R:{rv:.4f}  B:{bv:.4f}",
                                    (10, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1)
                        dy += 16

                # ─── คำแนะนำ ───
                dy += 20
                cv2.putText(diff_panel, "How to test:", (10, dy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 0), 1)
                dy += 18
                cv2.putText(diff_panel, "1. Real face -> note values", (10, dy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
                dy += 15
                cv2.putText(diff_panel, "2. Phone screen -> note values", (10, dy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
                dy += 15
                cv2.putText(diff_panel, "3. Find threshold between them", (10, dy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
                dy += 15
                cv2.putText(diff_panel, "4. Blur should widen the gap", (10, dy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

                # FPS + kernel info
                cv2.putText(diff_panel, f"FPS:{fps:.0f}  Kernel:{GAUSSIAN_KERNEL}  [+/-]",
                            (10, panel_h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

                combined = np.hstack([panel_raw, display, panel_blur, diff_panel])
            else:
                empty = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
                combined = np.hstack([empty, display, empty, empty])
        else:
            # ไม่เจอหน้า
            cv2.putText(display, "No face", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
            empty = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
            cv2.putText(empty, "Waiting for face...", (10, panel_h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            combined = np.hstack([empty, display, empty.copy(), empty.copy()])

        cv2.imshow("Gaussian + Fourier Test", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("s"):
            fname = f"gauss_fourier_{int(time.time())}.png"
            cv2.imwrite(fname, combined)
            print(f"Screenshot: {fname}")
        elif key == ord("+") or key == ord("="):
            GAUSSIAN_KERNEL = min(31, GAUSSIAN_KERNEL + 2)
            print(f"Kernel: {GAUSSIAN_KERNEL}")
        elif key == ord("-") or key == ord("_"):
            GAUSSIAN_KERNEL = max(3, GAUSSIAN_KERNEL - 2)
            print(f"Kernel: {GAUSSIAN_KERNEL}")
        elif key == ord("r"):
            stats = {k: [] for k in stats}
            print("Stats reset")

    cap.release()
    cv2.destroyAllWindows()

    # ─── สรุป ───
    print()
    print("=" * 60)
    print("  Summary — Average values")
    print("=" * 60)
    if len(stats["raw_hi"]) > 0:
        print(f"  {'Metric':<12} {'Raw':>10} {'Blurred':>10} {'Diff':>10}")
        print(f"  {'-'*42}")
        for name, rk, bk in [("HiFreq","raw_hi","blur_hi"),
                              ("Moire","raw_moire","blur_moire"),
                              ("Flatness","raw_flat","blur_flat")]:
            rv = np.mean(stats[rk])
            bv = np.mean(stats[bk])
            print(f"  {name:<12} {rv:>10.4f} {bv:>10.4f} {bv-rv:>+10.4f}")
    else:
        print("  ไม่มีข้อมูล")
    print("=" * 60)
    print()
    print("  ถ้าค่า Diff ของ HiFreq/Moire เป็น ลบ")
    print("  → Gaussian ช่วยลด noise ของจอได้")
    print("  → ใช้เป็น preprocessing ก่อน Fourier ได้ผลดี")
    print()


if __name__ == "__main__":
    main()
