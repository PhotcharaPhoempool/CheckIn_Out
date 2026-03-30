"""
camera.py (Windows Version) — กล้องแบบ threaded
รองรับทั้ง USB/local และ IP camera (RTSP/HTTP)
ความต่างจาก Linux: ใช้ CAP_DSHOW แทน CAP_V4L2
"""

import os
import time
import cv2
from threading import Thread, Lock


class ThreadedCamera:
    """
    อ่านเฟรมจากกล้องใน background thread
    รองรับ:
      - กล้อง USB/local  → ThreadedCamera(0)
      - IP camera (RTSP) → ThreadedCamera("rtsp://...")
      - IP camera (HTTP) → ThreadedCamera("http://...")
    """

    def __init__(self, src=0, reconnect_delay: float = 2.0):
        self._src = src
        self._is_url = isinstance(src, str)
        self._reconnect_delay = reconnect_delay
        self._lock = Lock()
        self._stopped = False
        self._ret = False
        self._frame = None

        if self._is_url:
            os.environ.setdefault(
                "OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp"
            )

        self.cap = self._open()
        self._ret, self._frame = self.cap.read()

        Thread(target=self._reader_loop, daemon=True).start()

    def _open(self) -> cv2.VideoCapture:
        if self._is_url:
            cap = cv2.VideoCapture(self._src, cv2.CAP_FFMPEG)
        else:
            # Windows: ลอง default ก่อน แล้ว DirectShow
            cap = cv2.VideoCapture(self._src)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self._src, cv2.CAP_DSHOW)

        if not cap.isOpened():
            raise RuntimeError(f"เปิดกล้องไม่ได้: src={self._src}")

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _reader_loop(self):
        while not self._stopped:
            ret, frame = self.cap.read()

            if not ret:
                if not self._is_url:
                    with self._lock:
                        self._ret = False
                    break

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

    def read(self):
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
