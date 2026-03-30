@echo off
:: ============================================================
:: run.bat — เปิดระบบตรวจใบหน้า (Windows)
:: ============================================================
:: ใช้งาน:
::   run.bat              ← ค่า default (ไม่มี profile)
::   run.bat cam_main     ← กล้อง IP หลัก
::   run.bat cam_usb      ← กล้อง USB / Webcam
::   run.bat test         ← โหมดทดสอบ 60 วินาที
:: ============================================================

cd /d "%~dp0"

:: ตรวจสอบว่ามี Python ไหม
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] ไม่พบ Python — ติดตั้ง Python 3.11+ ก่อน
    pause
    exit /b 1
)

:: ตรวจว่ามี encodings.pkl ไหม (อยู่ที่ root)
if not exist "..\encodings.pkl" (
    echo [WARN] ไม่พบ encodings.pkl — รัน encode_faces_arcface.py ก่อน
    echo        cd .. ^&^& python encode_faces_arcface.py
    pause
    exit /b 1
)

echo [START] กำลังเปิดระบบ...
if "%~1"=="" (
    python main.py
) else (
    python main.py %1
)

if errorlevel 1 (
    echo.
    echo [ERROR] ระบบปิดผิดปกติ — ดู error ด้านบน
    pause
)
