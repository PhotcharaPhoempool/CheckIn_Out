@echo off
:: ============================================================
:: install.bat — ติดตั้ง dependencies สำหรับ Windows
:: ============================================================
:: รันครั้งแรกครั้งเดียว (ใช้เวลา 5-15 นาที)
:: ต้องมี Python 3.11+ และ pip อยู่แล้ว
:: ============================================================

cd /d "%~dp0"

echo ============================================================
echo   Face Attendance System — Windows Installer
echo ============================================================
echo.

:: ตรวจ Python
python --version
if errorlevel 1 (
    echo [ERROR] ไม่พบ Python กรุณาติดตั้ง Python 3.11+ จาก python.org
    pause
    exit /b 1
)

echo.
echo [1/5] อัปเกรด pip...
python -m pip install --upgrade pip

echo.
echo [2/5] ติดตั้ง packages หลัก...
pip install numpy opencv-python pillow mediapipe

echo.
echo [3/5] ติดตั้ง InsightFace...
pip install insightface

echo.
echo [4/5] ติดตั้ง onnxruntime-gpu (ถ้ามี GPU) หรือ onnxruntime (CPU)...
echo       [ถ้ามี NVIDIA GPU ให้กด G, ถ้าไม่มีกด C]
choice /c GC /m "เลือก GPU(G) หรือ CPU(C)"
if errorlevel 2 (
    echo [4/5] ติดตั้ง onnxruntime (CPU)...
    pip install onnxruntime
) else (
    echo [4/5] ติดตั้ง onnxruntime-gpu...
    pip install onnxruntime-gpu
)

echo.
echo [5/5] ติดตั้ง psycopg (PostgreSQL)...
pip install "psycopg[binary]"

echo.
echo ============================================================
echo   ติดตั้งเสร็จแล้ว!
echo.
echo   ขั้นตอนต่อไป:
echo   1. แก้ไข Win_Ver\profiles\cam_main.py ให้ตรงกับ IP กล้อง
echo   2. แก้ไข Win_Ver\config.py ส่วน DB_CONFIG ให้ตรงกับ database
echo   3. รัน: run.bat cam_main
echo ============================================================
pause
