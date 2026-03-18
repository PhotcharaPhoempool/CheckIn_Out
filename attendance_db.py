from db import get_connection

def get_employee_by_name(full_name):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, employee_code, full_name
                FROM employees
                WHERE full_name = %s
                LIMIT 1
            """, (full_name,))
            return cur.fetchone()

def already_marked_today(employee_id, status):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 1
                FROM attendance_logs
                WHERE employee_id = %s
                  AND DATE(check_time) = CURRENT_DATE
                  AND status = %s
                LIMIT 1
            """, (employee_id, status))
            return cur.fetchone() is not None

def has_checked_in_today(employee_id):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 1
                FROM attendance_logs
                WHERE employee_id = %s
                  AND DATE(check_time) = CURRENT_DATE
                  AND status = 'IN'
                LIMIT 1
            """, (employee_id,))
            return cur.fetchone() is not None

def mark_attendance(employee_id, status, camera_name=None):
    try:
        if already_marked_today(employee_id, status):
            print(f"วันนี้บันทึก {status} แล้ว")
            return False

        if status == "OUT" and not has_checked_in_today(employee_id):
            print("ยังไม่มีข้อมูลเข้างานวันนี้ จึงเช็กเอาต์ไม่ได้")
            return False

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO attendance_logs (employee_id, status, camera_name)
                    VALUES (%s, %s, %s)
                """, (employee_id, status, camera_name))
            conn.commit()

        print(f"บันทึกเวลา {status} สำเร็จ")
        return True

    except Exception as e:
        print("เกิดข้อผิดพลาด:", e)
        return False