"""
attendance_db.py — เชื่อมต่อฐานข้อมูลพนักงาน + บันทึกเวลา
============================================================
โครงสร้างตาราง:
  employees:
    employee_code  VARCHAR(20) PK   ← เลขบัตรประจำตัว (ไม่ซ้ำ)
    title          VARCHAR(20)       ← คำนำหน้า (นาย/นาง/นางสาว)
    first_name     VARCHAR(100)      ← ชื่อ
    last_name      VARCHAR(100)      ← นามสกุล
    face_label     VARCHAR(100)      ← ชื่อโฟลเดอร์ใน dataset (match กับ face recognition)

  attendance_logs:
    id             SERIAL PK
    employee_code  VARCHAR(20) FK    ← เลขบัตร
    status         'IN' / 'OUT'
    camera_name    VARCHAR(50)
    check_time     TIMESTAMP
"""

from db import get_connection


# ─── ค้นหาพนักงาน ──────────────────────────

def get_employee_by_name(face_label: str):
    """
    ค้นหาพนักงานจากชื่อที่ face recognition จำได้ (face_label = ชื่อโฟลเดอร์)
    
    Returns: (employee_code, title, first_name, last_name, face_label) หรือ None
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT employee_code, title, first_name, last_name, face_label
                FROM employees
                WHERE face_label = %s AND is_active = TRUE
                LIMIT 1
            """, (face_label,))
            return cur.fetchone()


def get_employee_by_code(employee_code: str):
    """
    ค้นหาพนักงานจากเลขบัตรประจำตัว
    
    Returns: (employee_code, title, first_name, last_name, face_label) หรือ None
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT employee_code, title, first_name, last_name, face_label
                FROM employees
                WHERE employee_code = %s AND is_active = TRUE
                LIMIT 1
            """, (employee_code,))
            return cur.fetchone()


def get_display_name(employee_row) -> str:
    """
    สร้างชื่อแสดงผลจากข้อมูลพนักงาน
    เช่น "นาย สมชาย ใจดี"
    
    Args:
        employee_row: tuple (employee_code, title, first_name, last_name, face_label)
    """
    if not employee_row:
        return "Unknown"
    _, title, first_name, last_name, _ = employee_row
    parts = [p for p in [title, first_name, last_name] if p]
    return " ".join(parts)


# ─── ตรวจสอบซ้ำ ────────────────────────────

def already_marked_today(employee_code: str, status: str) -> bool:
    """ตรวจว่าวันนี้บันทึก IN/OUT แล้วหรือยัง"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 1
                FROM attendance_logs
                WHERE employee_code = %s
                  AND DATE(check_time) = CURRENT_DATE
                  AND status = %s
                LIMIT 1
            """, (employee_code, status))
            return cur.fetchone() is not None


def has_checked_in_today(employee_code: str) -> bool:
    """ตรวจว่าวันนี้มี IN แล้วหรือยัง"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 1
                FROM attendance_logs
                WHERE employee_code = %s
                  AND DATE(check_time) = CURRENT_DATE
                  AND status = 'IN'
                LIMIT 1
            """, (employee_code,))
            return cur.fetchone() is not None


# ─── บันทึกเวลา ────────────────────────────

def mark_attendance(employee_code: str, status: str,
                    camera_name: str = None, check_time=None) -> bool:
    """
    บันทึก IN หรือ OUT
    
    Args:
        employee_code: เลขบัตรประจำตัว (PK)
        status: "IN" หรือ "OUT"
        camera_name: ชื่อกล้อง (optional)
        check_time: เวลาที่ต้องการบันทึก (optional, default=now)
    
    Returns: True ถ้าสำเร็จ, False ถ้าซ้ำหรือ error
    """
    try:
        if already_marked_today(employee_code, status):
            print(f"[DB] วันนี้บันทึก {status} ({employee_code}) แล้ว — ข้าม")
            return False

        if status == "OUT" and not has_checked_in_today(employee_code):
            print(f"[DB] {employee_code} ยังไม่มี IN วันนี้ — ไม่บันทึก OUT")
            return False

        with get_connection() as conn:
            with conn.cursor() as cur:
                if check_time is not None:
                    cur.execute("""
                        INSERT INTO attendance_logs
                            (employee_code, status, camera_name, check_time)
                        VALUES (%s, %s, %s, %s)
                    """, (employee_code, status, camera_name, check_time))
                else:
                    cur.execute("""
                        INSERT INTO attendance_logs
                            (employee_code, status, camera_name)
                        VALUES (%s, %s, %s)
                    """, (employee_code, status, camera_name))
            conn.commit()

        print(f"[DB] บันทึก {status} สำเร็จ ({employee_code})")
        return True

    except Exception as e:
        print(f"[DB ERROR] mark_attendance({employee_code}, {status}): {e}")
        import traceback
        traceback.print_exc()
        return False