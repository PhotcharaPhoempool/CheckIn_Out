"""
camera.py — กล้องแบบ threaded (รองรับทั้ง USB/local และ IP camera)
"""

import os
import time
import cv2
from threading import Thread, Lock


class ThreadedCamera:
    """
    อ่านเฟรมจากกล้องใน background thread
    รองรับ:
      - กล้อง USB/local  → ThreadedCamera(0) หรือ ThreadedCamera(1)
      - IP camera (RTSP) → ThreadedCamera("rtsp://...")
      - IP camera (HTTP) → ThreadedCamera("http://...")

    สำหรับ IP camera จะ reconnect อัตโนมัติเมื่อสัญญาณหาย
    """

    def __init__(self, src=0, reconnect_delay: float = 2.0):
        self._src = src
        self._is_url = isinstance(src, str)
        self._reconnect_delay = reconnect_delay
        self._lock = Lock()
        self._stopped = False
        self._ret = False
        self._frame = None

        # สำหรับ RTSP: บังคับใช้ TCP transport (เสถียรกว่า UDP บน LAN)
        if self._is_url:
            os.environ.setdefault(
                "OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp"
            )

        self.cap = self._open()
        self._ret, self._frame = self.cap.read()

        Thread(target=self._reader_loop, daemon=True).start()

    # ──────────────────────────────────────────
    def _open(self) -> cv2.VideoCapture:
        """เปิด capture — เลือก backend ตามชนิดของ source"""
        if self._is_url:
            cap = cv2.VideoCapture(self._src, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(self._src)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self._src, cv2.CAP_V4L2)

        if not cap.isOpened():
            raise RuntimeError(f"เปิดกล้องไม่ได้: src={self._src}")

        # buffer=1 → ลด latency (ไม่สะสมเฟรมเก่า)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    # ──────────────────────────────────────────
    def _reader_loop(self):
        """อ่านเฟรมวนลูป — reconnect อัตโนมัติถ้าเป็น IP camera"""
        while not self._stopped:
            ret, frame = self.cap.read()

            if not ret:
                if not self._is_url:
                    # กล้อง local หาย → หยุดเลย
                    with self._lock:
                        self._ret = False
                    break

                # IP camera หาย → รอแล้ว reconnect
                print(f"[CAM] สัญญาณขาด — reconnect ใน {self._reconnect_delay:.0f}s...")
                time.sleep(self._reconnect_delay)
                try:
                    self.cap.release()
                    self.cap = self._open()
                    print("[CAM] reconnect สำเร็จ")
                except Exception as e:
                    print(f"[CAM] reconnect ล้มเหลว: {e}")
                continue

            with self._lock:
                self._ret, self._frame = ret, frame

    # ──────────────────────────────────────────
    def read(self):
        """คืนเฟรมล่าสุด (ไม่ block)"""
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ret, self._frame.copy()

    def release(self):
        self._stopped = True
        self.cap.release()

    @property
    def width(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
