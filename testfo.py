"""
test_fourier_spoof.py — ทดสอบ Fourier Transform แยกหน้าจริง vs หน้าจอ
=========================================================================
วิธีใช้:
  python test_fourier_spoof.py              ← ใช้กล้อง index 0
  python test_fourier_spoof.py --cam 1      ← ใช้กล้อง index 1
  python test_fourier_spoof.py --cam 2      ← ใช้กล้อง index 2

วิธีทดสอบ:
  1. หันหน้าเข้ากล้อง → ดูว่าระบบบอก "REAL" หรือ "SCREEN"
  2. เอามือถือที่เปิดรูป/วิดีโอหน้าคนมาส่องกล้อง → ดูว่าจับได้ไหม
  3. ลองจอ laptop / tablet ด้วย

หน้าจอจะแสดง:
  - ภาพจากกล้อง + กรอบหน้า
  - Frequency spectrum (Fourier)
  - ค่าตัวเลข 3 ตัว + ผลตัดสิน REAL/SCREEN
  - กราฟ bar แสดง score แบบ real-time

กด q เพื่อออก | กด s เพื่อ screenshot
"""

import cv2
import numpy as np
import argparse
import time

# ─── ค่า threshold (ปรับได้) ────────────────
# ค่าเริ่มต้นเหล่านี้เป็นค่าประมาณ — ต้องปรับตามกล้องจริง
# รันแล้วดูค่าที่แสดง แล้วปรับให้เหมาะ

HIGHFREQ_ENERGY_MAX   = 0.42    # ← ค่านี้ OK แล้ว (ทั้งคู่ 0.002)
MOIRÉ_PEAK_MAX        = 28.0    # ← เดิม 2.5 เปลี่ยนเป็น 28 (อยู่ระหว่าง 23.88 กับ 31.07)
SPECTRAL_FLATNESS_MIN = 0.33    # ← เดิม 0.55 เปลี่ยนเป็น 0.33 (อยู่ระหว่าง 0.305 กับ 0.361)
# ────────────────────────────────────────────


def analyze_fourier(face_bgr: np.ndarray) -> dict:
    """
    วิเคราะห์ Fourier Transform ของใบหน้า

    Returns dict:
      high_energy   : สัดส่วน high-frequency energy (จอสูง, หน้าจริงต่ำ)
      moire_score   : ค่า moiré peak (จอสูง)
      flatness      : spectral flatness (หน้าจริงสูง, จอต่ำ)
      is_screen     : True ถ้าคิดว่าเป็นจอ
      checks        : จำนวนเกณฑ์ที่ fail (0=real, 2-3=screen)
    """
    # แปลงเป็น grayscale แล้ว resize ให้คงที่
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))

    # ─── FFT ───
    f_transform = np.fft.fft2(gray.astype(np.float32))
    f_shift     = np.fft.fftshift(f_transform)
    magnitude   = np.abs(f_shift)

    # log scale (หลีก log(0))
    mag_log = np.log1p(magnitude)

    # ─── 1. High-Frequency Energy Ratio ───
    # หน้าจริง: พลังงานส่วนใหญ่อยู่ low-freq (center)
    # จอ: มี pixel grid → high-freq energy สูง
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 4  # วงกลม low-freq zone

    # สร้าง mask แยก low/high freq
    Y, X = np.ogrid[:h, :w]
    low_mask  = ((X - cx)**2 + (Y - cy)**2) <= radius**2
    high_mask = ~low_mask

    total_energy = np.sum(magnitude**2) + 1e-10
    high_energy  = np.sum(magnitude[high_mask]**2)
    high_ratio   = high_energy / total_energy

    # ─── 2. Moiré Detection ───
    # จอมีลาย moiré → spike ที่ frequency เฉพาะ
    # วัดจากอัตราส่วน peak vs median ใน high-freq zone
    high_vals = magnitude[high_mask]
    if len(high_vals) > 0 and np.median(high_vals) > 0:
        moire_score = np.max(high_vals) / np.median(high_vals)
    else:
        moire_score = 0.0

    # ─── 3. Spectral Flatness ───
    # หน้าจริง: spectrum ราบ (ลดตาม 1/f ธรรมชาติ)
    # จอ: มี spike → flatness ต่ำ
    # flatness = geometric_mean / arithmetic_mean
    flat_mag = magnitude.flatten()
    flat_mag = flat_mag[flat_mag > 0]  # ตัด 0
    if len(flat_mag) > 0:
        log_mean = np.mean(np.log(flat_mag + 1e-10))
        geo_mean = np.exp(log_mean)
        ari_mean = np.mean(flat_mag)
        flatness = geo_mean / ari_mean if ari_mean > 0 else 0
    else:
        flatness = 0.0

    # ─── ตัดสิน ───
    checks_failed = 0
    if high_ratio > HIGHFREQ_ENERGY_MAX:
        checks_failed += 1
    if moire_score > MOIRÉ_PEAK_MAX:
        checks_failed += 1
    if flatness < SPECTRAL_FLATNESS_MIN:
        checks_failed += 1

    return {
        "high_energy":  high_ratio,
        "moire_score":  moire_score,
        "flatness":     flatness,
        "is_screen":    checks_failed >= 2,
        "checks_failed": checks_failed,
        "magnitude_log": mag_log,   # สำหรับ visualize
    }


def draw_bar(img, x, y, w, h, value, max_val, threshold,
             label, color_ok, color_fail, above_is_bad=True):
    """วาด bar chart + threshold line"""
    # พื้นหลัง
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), -1)

    # ค่า bar
    ratio = min(value / max_val, 1.0)
    bar_h = int(h * ratio)

    if above_is_bad:
        is_bad = value > threshold
    else:
        is_bad = value < threshold

    color = color_fail if is_bad else color_ok
    cv2.rectangle(img, (x, y + h - bar_h), (x + w, y + h), color, -1)

    # threshold line
    th_y = y + h - int(h * (threshold / max_val))
    cv2.line(img, (x - 3, th_y), (x + w + 3, th_y), (0, 255, 255), 2)

    # label
    cv2.putText(img, label, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(img, f"{value:.3f}", (x, y + h + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (180, 180, 180), 1)

    # pass/fail
    status = "FAIL" if is_bad else "OK"
    s_color = (0, 0, 255) if is_bad else (0, 255, 0)
    cv2.putText(img, status, (x + 2, y + h + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, s_color, 1)


def create_spectrum_image(mag_log, size=200):
    """สร้างภาพ spectrum สำหรับแสดงผล"""
    # normalize to 0-255
    norm = cv2.normalize(mag_log, None, 0, 255, cv2.NORM_MINMAX)
    spec = norm.astype(np.uint8)
    spec = cv2.resize(spec, (size, size))
    # colormap
    spec_color = cv2.applyColorMap(spec, cv2.COLORMAP_JET)
    return spec_color


def main():
    parser = argparse.ArgumentParser(description="Fourier Anti-Spoof Test")
    parser.add_argument("--cam", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    # เปิดกล้อง
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"เปิดกล้อง {args.cam} ไม่ได้!")
        return

    print("=" * 55)
    print("  Fourier Anti-Spoof Test")
    print("=" * 55)
    print(f"  กล้อง: {args.cam}")
    print(f"  Threshold:")
    print(f"    High-Freq Energy MAX : {HIGHFREQ_ENERGY_MAX}")
    print(f"    Moiré Peak MAX       : {MOIRÉ_PEAK_MAX}")
    print(f"    Spectral Flatness MIN: {SPECTRAL_FLATNESS_MIN}")
    print("=" * 55)
    print("  [q] ออก   [s] screenshot   [+/-] ปรับ threshold")
    print()

    # ใช้ face_recognition หาหน้า (ถ้ามี) หรือใช้ Haar cascade
    try:
        import face_recognition
        use_fr = True
        print("  ใช้ face_recognition library ตรวจจับหน้า")
    except ImportError:
        use_fr = False
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        print("  ใช้ Haar cascade ตรวจจับหน้า (ติดตั้ง face_recognition จะแม่นกว่า)")

    print()
    frame_count = 0
    fps_timer = time.time()
    fps = 0

    # สะสมผลลัพธ์
    history_real   = 0
    history_screen = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            fps = frame_count / (now - fps_timer)
            frame_count = 0
            fps_timer = now

        # ─── หาหน้า ───
        display = frame.copy()
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if use_fr:
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb)
            faces = []
            for (t, r, b, l) in locs:
                faces.append((l*4, t*4, (r-l)*4, (b-t)*4))
        else:
            detected = face_cascade.detectMultiScale(gray_full, 1.3, 5)
            faces = [(x, y, w, h) for (x, y, w, h) in detected]

        # ─── วิเคราะห์แต่ละหน้า ───
        result = None
        for (x, y, w, h) in faces:
            pad = 10
            fh, fw = frame.shape[:2]
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(fw, x + w + pad)
            y2 = min(fh, y + h + pad)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0 or face_crop.shape[0] < 30 or face_crop.shape[1] < 30:
                continue

            result = analyze_fourier(face_crop)

            # ─── วาดกรอบ + ผล ───
            if result["is_screen"]:
                color = (0, 0, 255)     # แดง = SCREEN
                label = "SCREEN"
                history_screen += 1
            else:
                color = (0, 255, 0)     # เขียว = REAL
                label = "REAL"
                history_real += 1

            cv2.rectangle(display, (x, y), (x+w, y+h), color, 3)

            # label ใหญ่
            cv2.rectangle(display, (x, y-40), (x+130, y), color, -1)
            cv2.putText(display, label, (x+5, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # ค่าตัวเลข
            cv2.putText(display, f"HiFreq: {result['high_energy']:.3f}", (x, y+h+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.putText(display, f"Moire:  {result['moire_score']:.2f}", (x, y+h+40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.putText(display, f"Flat:   {result['flatness']:.3f}", (x, y+h+60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.putText(display, f"Fail: {result['checks_failed']}/3", (x, y+h+80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            break  # ใช้หน้าแรกที่เจอ

        # ─── สร้าง panel ขวา ───
        panel_w = 280
        panel = np.zeros((frame.shape[0], panel_w, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)

        # Header
        cv2.rectangle(panel, (0, 0), (panel_w, 40), (50, 50, 50), -1)
        cv2.putText(panel, "Fourier Analysis", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if result:
            # Spectrum image
            spec_img = create_spectrum_image(result["magnitude_log"], size=140)
            py = 55
            px = (panel_w - 140) // 2
            panel[py:py+140, px:px+140] = spec_img
            cv2.putText(panel, "Frequency Spectrum", (px, py-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

            # Bar charts
            bar_y = 220
            bar_h = 100
            bar_w = 55
            gap   = 25

            # Bar 1: High-Freq Energy
            bx1 = 30
            draw_bar(panel, bx1, bar_y, bar_w, bar_h,
                     result["high_energy"], 1.0, HIGHFREQ_ENERGY_MAX,
                     "HiFreq", (0,200,0), (0,0,200), above_is_bad=True)

            # Bar 2: Moiré Score
            bx2 = bx1 + bar_w + gap
            draw_bar(panel, bx2, bar_y, bar_w, bar_h,
                     result["moire_score"], 8.0, MOIRÉ_PEAK_MAX,
                     "Moire", (0,200,0), (0,0,200), above_is_bad=True)

            # Bar 3: Flatness
            bx3 = bx2 + bar_w + gap
            draw_bar(panel, bx3, bar_y, bar_w, bar_h,
                     result["flatness"], 1.0, SPECTRAL_FLATNESS_MIN,
                     "Flatness", (0,200,0), (0,0,200), above_is_bad=False)

            # ─── ผลรวม ───
            verdict_y = bar_y + bar_h + 55
            if result["is_screen"]:
                cv2.rectangle(panel, (10, verdict_y-25), (panel_w-10, verdict_y+15),
                              (0, 0, 180), -1)
                cv2.putText(panel, "SCREEN DETECTED", (20, verdict_y+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            else:
                cv2.rectangle(panel, (10, verdict_y-25), (panel_w-10, verdict_y+15),
                              (0, 140, 0), -1)
                cv2.putText(panel, "REAL FACE", (55, verdict_y+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # ─── สถิติ ───
            stat_y = verdict_y + 40
            total = history_real + history_screen
            cv2.putText(panel, f"Stats ({total} frames):", (10, stat_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            cv2.putText(panel, f"  REAL:   {history_real}", (10, stat_y+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
            cv2.putText(panel, f"  SCREEN: {history_screen}", (10, stat_y+40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
            if total > 0:
                pct = history_real / total * 100
                cv2.putText(panel, f"  Real%:  {pct:.1f}%", (10, stat_y+60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)

        else:
            cv2.putText(panel, "No face detected", (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            cv2.putText(panel, "Show your face", (40, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
            cv2.putText(panel, "or phone screen", (35, 175),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

        # FPS
        cv2.putText(panel, f"FPS: {fps:.0f}", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

        # ─── Threshold info ───
        info_y = frame.shape[0] - 80
        cv2.putText(panel, "Thresholds:", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120, 120, 120), 1)
        cv2.putText(panel, f"  HiFreq < {HIGHFREQ_ENERGY_MAX}", (10, info_y+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120, 120, 120), 1)
        cv2.putText(panel, f"  Moire  < {MOIRÉ_PEAK_MAX}", (10, info_y+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120, 120, 120), 1)
        cv2.putText(panel, f"  Flat   > {SPECTRAL_FLATNESS_MIN}", (10, info_y+45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120, 120, 120), 1)

        # ─── รวมแสดงผล ───
        combined = np.hstack([display, panel])
        cv2.imshow("Fourier Anti-Spoof Test", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("s"):
            fname = f"fourier_test_{int(time.time())}.png"
            cv2.imwrite(fname, combined)
            print(f"Screenshot saved: {fname}")
        elif key == ord("r"):
            # Reset stats
            history_real = 0
            history_screen = 0
            print("Stats reset")

    cap.release()
    cv2.destroyAllWindows()

    # ─── สรุปผล ───
    print()
    print("=" * 40)
    print("  ผลทดสอบสรุป")
    print("=" * 40)
    total = history_real + history_screen
    if total > 0:
        print(f"  REAL:   {history_real} ({history_real/total*100:.1f}%)")
        print(f"  SCREEN: {history_screen} ({history_screen/total*100:.1f}%)")
        print(f"  Total:  {total} frames")
    else:
        print("  ไม่มีข้อมูล")
    print("=" * 40)


if __name__ == "__main__":
    main()