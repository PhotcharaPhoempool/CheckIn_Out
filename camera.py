"""
camera.py — กล้องแบบ threaded (อ่านเฟรมไม่ block main loop)
"""

import cv2
from threading import Thread, Lock


class ThreadedCamera:
    """
    อ่านเฟรมจากกล้องใน background thread
    ลด latency เพราะ main loop ไม่ต้องรอ I/O ของกล้อง

    ใช้งาน:
        cam = ThreadedCamera(src=0)
        ret, frame = cam.read()
        cam.release()
    """

    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"เปิดกล้องไม่ได้: src={src}")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._lock = Lock()
        self._ret, self._frame = self.cap.read()
        self._stopped = False

        Thread(target=self._reader_loop, daemon=True).start()

    def _reader_loop(self):
        """อ่านเฟรมจากกล้องวนลูปไม่หยุด"""
        while not self._stopped:
            ret, frame = self.cap.read()
            with self._lock:
                self._ret, self._frame = ret, frame

    def read(self):
        """อ่านเฟรมล่าสุด (ไม่ block)"""
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