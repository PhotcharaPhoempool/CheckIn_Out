# FaceReg — Face Attendance System

## ผู้ใช้
- นักศึกษาฝึกงาน พัฒนาระบบ Face Attendance ด้วย Python/OpenCV/InsightFace
- ผ่านมาแล้วหลาย version (v3–v7) — มีประสบการณ์กับ codebase นี้พอสมควร
- พูดคุยภาษาไทยเป็นหลัก — ตอบกลับภาษาไทยเสมอ

## Stack
- **Face recognition:** InsightFace (buffalo_l, ArcFace 512d) + ONNX Runtime (GPU)
- **Camera:** ThreadedCamera รองรับทั้ง USB index และ IP camera (RTSP)
- **Anti-spoofing:** MediaPipe Hands (finger challenge) + MiniFASNet
- **Database:** SQLite ผ่าน attendance_db.py / db.py
- **UI:** OpenCV วาด overlay, panel ขวา, HUD

## ไฟล์หลัก
| ไฟล์ | หน้าที่ |
|---|---|
| `main.py` | Main loop — detect, identify, liveness check |
| `config.py` | ตั้งค่าทั้งหมดในที่เดียว (แก้ที่นี่ที่เดียว) |
| `liveness_engine.py` | Anti-spoofing pipeline ทุกด่าน |
| `session_manager.py` | จัดการ session / check-in / check-out |
| `ui_renderer.py` | วาด UI ทุกอย่าง (face box, oval guide, panel, HUD) |
| `camera.py` | ThreadedCamera |
| `attendance_db.py` | บันทึก/ดึงข้อมูลการลงเวลา |
| `encode_faces_arcface.py` | สร้าง encodings.pkl จากรูปใน known_faces/ |
| `profiles/` | Config profiles สลับได้ด้วย argument |
| `Win_Ver/` | Windows version ของโปรเจกต์ — **แก้เฉพาะเมื่อสั่งเท่านั้น** โดย default แก้ Linux version ก่อน |

## Anti-Spoofing Pipeline
1. **Landmark Depth** — ตรวจความลึก 3D จาก 68-point landmarks
2. **Micro-Motion** — ตรวจการขยับเล็กน้อย
3. **Blink Detection** — EAR จาก 68-point landmarks
4. **Texture** — LBP + Laplacian + Chroma
5. **Screen Border** — ปิดอยู่ (false positive จากกล้อง wide-angle fisheye)
6. **Finger Challenge** — MediaPipe Hands ให้ชูนิ้ว 2 ชุดต่างกัน
7. **MiniFASNet (FAS)** — AI model ตรวจ spoof

## IP Camera
- RTSP: `rtsp://admin:@dmin123456@192.168.1.13:554/unicast/c1/s0/live`
- `CAMERA_FLIP = False`

## Profile System
```bash
python main.py cam_main    # โหลด profiles/cam_main.py
python main.py test        # โหลด profiles/test.py
FACE_PROFILE=cam_usb python main.py
```
