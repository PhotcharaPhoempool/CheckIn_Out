"""
test_minifas_spoof.py — ทดสอบ MiniFASNet (Silent-Face-Anti-Spoofing)
=====================================================================
ใช้ DeepFace library ที่มี MiniFASNet built-in

ติดตั้ง:
  pip install deepface tf-keras opencv-python

วิธีใช้:
  python test_minifas_spoof.py              ← กล้อง 0
  python test_minifas_spoof.py --cam 1      ← กล้อง 1

วิธีทดสอบ:
  1. หันหน้าจริงเข้ากล้อง   → ควรขึ้น REAL (เขียว)
  2. เอามือถือเปิดรูปมาส่อง  → ควรขึ้น SPOOF (แดง)
  3. เปิดวิดีโอมาส่อง       → ควรขึ้น SPOOF (แดง)
  4. ลองจอ laptop          → ควรขึ้น SPOOF (แดง)

ปุ่มกด:
  q = ออก  |  s = screenshot  |  r = reset สถิติ
"""

import cv2
import numpy as np
import argparse
import time
import sys


def check_dependencies():
    """ตรวจว่ามี library ที่ต้องการ"""
    missing = []
    try:
        import deepface
        print(f"  DeepFace version: {deepface.__version__}")
    except ImportError:
        missing.append("deepface")

    try:
        import torch
        print(f"  PyTorch version:  {torch.__version__}")
        print(f"  CUDA available:   {torch.cuda.is_available()}")
    except ImportError:
        # DeepFace ใช้ tf-keras เป็นหลัก, torch optional
        pass

    if missing:
        print(f"\n  ขาด library: {', '.join(missing)}")
        print(f"  ติดตั้ง: pip install {' '.join(missing)} tf-keras")
        return False
    return True


def analyze_spoof_deepface(frame, face_box=None):
    """
    ใช้ DeepFace ตรวจ anti-spoofing
    Returns: dict {is_real, score, time_ms} หรือ None ถ้าไม่เจอหน้า
    """
    from deepface import DeepFace

    start = time.time()

    try:
        results = DeepFace.extract_faces(
            img_path=frame,
            anti_spoofing=True,
            enforce_detection=True,
            detector_backend="opencv",   # เร็วสุด
        )

        elapsed = (time.time() - start) * 1000  # ms

        if not results:
            return None

        face = results[0]
        return {
            "is_real":  face.get("is_real", False),
            "score":    face.get("antispoof_score", 0.0),
            "time_ms":  elapsed,
            "box":      face.get("facial_area", {}),
        }

    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return {"is_real": None, "score": 0.0, "time_ms": elapsed, "error": str(e)}


def draw_confidence_bar(img, x, y, w, h, score, label):
    """วาด confidence bar"""
    # พื้นหลัง
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), -1)

    # ค่า bar
    bar_w = int(w * min(score, 1.0))

    # สี: เขียว (real) → แดง (spoof)
    if score > 0.5:
        r = int(255 * (1 - score) * 2)
        g = 200
    else:
        r = 200
        g = int(255 * score * 2)
    color = (0, g, r)

    cv2.rectangle(img, (x, y), (x + bar_w, y + h), color, -1)

    # threshold line ที่ 0.5
    th_x = x + w // 2
    cv2.line(img, (th_x, y - 2), (th_x, y + h + 2), (0, 255, 255), 2)

    # label
    cv2.putText(img, label, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    cv2.putText(img, f"{score:.3f}", (x + w + 5, y + h - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)


def main():
    parser = argparse.ArgumentParser(description="MiniFASNet Anti-Spoof Test")
    parser.add_argument("--cam", type=int, default=0, help="Camera index")
    parser.add_argument("--skip", type=int, default=3,
                        help="ตรวจทุกกี่เฟรม (default=3, สูง=เร็วขึ้น)")
    args = parser.parse_args()

    print("=" * 55)
    print("  MiniFASNet Anti-Spoof Test (via DeepFace)")
    print("=" * 55)

    if not check_dependencies():
        sys.exit(1)

    # Pre-load model (ครั้งแรกจะ download อัตโนมัติ)
    print("\n  กำลังโหลด model (ครั้งแรกจะ download)...")
    from deepface import DeepFace

    # Warm up — ให้ download model ก่อน
    try:
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        dummy[30:70, 30:70] = 128
        DeepFace.extract_faces(dummy, anti_spoofing=True, enforce_detection=False,
                               detector_backend="skip")
        print("  Model โหลดสำเร็จ!")
    except Exception as e:
        print(f"  Warning: warm-up: {e}")

    # เปิดกล้อง
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"  เปิดกล้อง {args.cam} ไม่ได้!")
        return

    print(f"\n  กล้อง: {args.cam}")
    print(f"  ตรวจทุก {args.skip} เฟรม (--skip)")
    print(f"  [q] ออก  [s] screenshot  [r] reset stats")
    print("=" * 55)

    frame_count = 0
    fps_cnt = 0
    fps_t = time.time()
    fps = 0

    last_result = None
    detect_time_avg = []

    # สถิติ
    stats_real = 0
    stats_spoof = 0
    stats_error = 0
    score_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        now = time.time()

        fps_cnt += 1
        if now - fps_t >= 1.0:
            fps = fps_cnt / (now - fps_t)
            fps_cnt = 0
            fps_t = now

        # ─── ตรวจ anti-spoofing ทุก N เฟรม ───
        if frame_count % args.skip == 0:
            result = analyze_spoof_deepface(frame)
            if result and "error" not in result:
                last_result = result
                detect_time_avg.append(result["time_ms"])
                if len(detect_time_avg) > 50:
                    detect_time_avg = detect_time_avg[-50:]

                if result["is_real"]:
                    stats_real += 1
                else:
                    stats_spoof += 1

                score_history.append(result["score"])
                if len(score_history) > 100:
                    score_history = score_history[-100:]

            elif result and "error" in result:
                stats_error += 1

        # ─── วาดผล ───
        display = frame.copy()

        if last_result and "error" not in last_result:
            is_real = last_result["is_real"]
            score = last_result["score"]
            t_ms = last_result["time_ms"]
            box = last_result.get("box", {})

            # วาดกรอบหน้า
            if box:
                x = box.get("x", 0)
                y = box.get("y", 0)
                w = box.get("w", 0)
                h = box.get("h", 0)

                if is_real:
                    color = (0, 220, 0)
                    label = f"REAL  {score:.1%}"
                else:
                    color = (0, 0, 220)
                    label = f"SPOOF  {score:.1%}"

                cv2.rectangle(display, (x, y), (x+w, y+h), color, 3)
                cv2.rectangle(display, (x, y-35), (x+160, y), color, -1)
                cv2.putText(display, label, (x+5, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ─── Panel ขวา ───
        panel_w = 300
        panel = np.zeros((frame.shape[0], panel_w, 3), dtype=np.uint8)
        panel[:] = (25, 25, 25)

        # Header
        cv2.rectangle(panel, (0, 0), (panel_w, 40), (50, 50, 50), -1)
        cv2.putText(panel, "MiniFASNet Analysis", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        py = 60

        if last_result and "error" not in last_result:
            is_real = last_result["is_real"]
            score = last_result["score"]

            # ─── Verdict ───
            if is_real:
                cv2.rectangle(panel, (10, py), (panel_w-10, py+40), (0, 140, 0), -1)
                cv2.putText(panel, "REAL FACE", (60, py+28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.rectangle(panel, (10, py), (panel_w-10, py+40), (0, 0, 160), -1)
                cv2.putText(panel, "SPOOF DETECTED", (30, py+28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            py += 55

            # ─── Score ───
            cv2.putText(panel, f"Confidence: {score:.4f}", (10, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            py += 8

            # Confidence bar
            draw_confidence_bar(panel, 10, py, panel_w-60, 18, score, "SPOOF <-- --> REAL")
            py += 40

            # ─── Speed ───
            avg_ms = np.mean(detect_time_avg) if detect_time_avg else 0
            cv2.putText(panel, f"Detection: {last_result['time_ms']:.0f}ms", (10, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            py += 18
            cv2.putText(panel, f"Average:   {avg_ms:.0f}ms", (10, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            py += 18
            cv2.putText(panel, f"Check every {args.skip} frames", (10, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
            py += 30

            # ─── Score History (mini chart) ───
            cv2.putText(panel, "Score history:", (10, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)
            py += 8

            chart_h = 60
            chart_w = panel_w - 20
            cv2.rectangle(panel, (10, py), (10+chart_w, py+chart_h), (35, 35, 35), -1)

            # threshold line
            th_y = py + chart_h // 2
            cv2.line(panel, (10, th_y), (10+chart_w, th_y), (0, 180, 180), 1)
            cv2.putText(panel, "0.5", (12, th_y-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 180, 180), 1)

            if len(score_history) > 1:
                for i in range(1, len(score_history)):
                    x1 = 10 + int((i-1) / len(score_history) * chart_w)
                    x2 = 10 + int(i / len(score_history) * chart_w)
                    y1 = py + chart_h - int(score_history[i-1] * chart_h)
                    y2 = py + chart_h - int(score_history[i] * chart_h)

                    c = (0, 200, 0) if score_history[i] > 0.5 else (0, 0, 200)
                    cv2.line(panel, (x1, y1), (x2, y2), c, 1)

            py += chart_h + 15

            # ─── Statistics ───
            total = stats_real + stats_spoof
            cv2.putText(panel, f"Statistics ({total} checks):", (10, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            py += 22

            cv2.putText(panel, f"  REAL:  {stats_real}", (10, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 0), 1)
            py += 18
            cv2.putText(panel, f"  SPOOF: {stats_spoof}", (10, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 200), 1)
            py += 18
            if stats_error > 0:
                cv2.putText(panel, f"  Error: {stats_error}", (10, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 150, 200), 1)
                py += 18

            if total > 0:
                pct = stats_real / total * 100
                cv2.putText(panel, f"  Real%: {pct:.1f}%", (10, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 0), 1)
                py += 18

            if score_history:
                avg_score = np.mean(score_history)
                cv2.putText(panel, f"  Avg score: {avg_score:.4f}", (10, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
                py += 25

            # ─── คำอธิบาย ───
            cv2.putText(panel, "Score > 0.5 = REAL", (10, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 180, 0), 1)
            py += 15
            cv2.putText(panel, "Score < 0.5 = SPOOF", (10, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 180), 1)

        else:
            cv2.putText(panel, "Waiting for face...", (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

            if last_result and "error" in last_result:
                cv2.putText(panel, "No face found", (30, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        # FPS
        cv2.putText(panel, f"FPS: {fps:.0f}", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        # ─── รวม ───
        combined = np.hstack([display, panel])
        cv2.imshow("MiniFASNet Anti-Spoof Test", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("s"):
            fname = f"minifas_test_{int(time.time())}.png"
            cv2.imwrite(fname, combined)
            print(f"Screenshot: {fname}")
        elif key == ord("r"):
            stats_real = 0
            stats_spoof = 0
            stats_error = 0
            score_history.clear()
            detect_time_avg.clear()
            print("Stats reset")

    cap.release()
    cv2.destroyAllWindows()

    # ─── สรุป ───
    print()
    print("=" * 55)
    print("  ผลทดสอบ MiniFASNet")
    print("=" * 55)
    total = stats_real + stats_spoof
    if total > 0:
        print(f"  REAL:       {stats_real:>5}  ({stats_real/total*100:.1f}%)")
        print(f"  SPOOF:      {stats_spoof:>5}  ({stats_spoof/total*100:.1f}%)")
        print(f"  Errors:     {stats_error:>5}")
        print(f"  Total:      {total:>5}")
        if score_history:
            print(f"  Avg score:  {np.mean(score_history):.4f}")
            print(f"  Min score:  {np.min(score_history):.4f}")
            print(f"  Max score:  {np.max(score_history):.4f}")
        if detect_time_avg:
            print(f"  Avg time:   {np.mean(detect_time_avg):.0f}ms")
    else:
        print("  ไม่มีข้อมูล")
    print("=" * 55)
    print()
    print("  ถ้าหน้าจริงได้ score > 0.8 และหน้าจอได้ < 0.3")
    print("  → MiniFASNet ใช้ได้ดีกับกล้องนี้ เพิ่มเข้าระบบเลย!")
    print()
    print("  ถ้า score ใกล้กัน (เช่น หน้าจริง 0.55, หน้าจอ 0.45)")
    print("  → กล้อง/แสงอาจไม่เหมาะ ลองปรับแสงหรือเปลี่ยนกล้อง")
    print()


if __name__ == "__main__":
    main()
