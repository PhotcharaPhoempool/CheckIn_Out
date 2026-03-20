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

def mark_attendance(employee_id, status, camera_name=None, check_time=None):
    try:
        if already_marked_today(employee_id, status):
            print(f"[DB] วันนี้บันทึก {status} (employee_id={employee_id}) แล้ว — ข้ามได้")
            return False

        if status == "OUT" and not has_checked_in_today(employee_id):
            print(f"[DB] employee_id={employee_id} ยังไม่มีข้อมูล IN วันนี้ ไม่สามารถบันทึก OUT")
            return False

        with get_connection() as conn:
            with conn.cursor() as cur:
                if check_time is not None:
                    cur.execute("""
                        INSERT INTO attendance_logs (employee_id, status, camera_name, check_time)
                        VALUES (%s, %s, %s, %s)
                    """, (employee_id, status, camera_name, check_time))
                else:
                    cur.execute("""
                        INSERT INTO attendance_logs (employee_id, status, camera_name)
                        VALUES (%s, %s, %s)
                    """, (employee_id, status, camera_name))
            conn.commit()

        print(f"[DB] บันทึก {status} สำเร็จ (employee_id={employee_id})")
        return True

    except Exception as e:
        print(f"[DB ERROR] mark_attendance({employee_id}, {status}): {e}")
        import traceback
        traceback.print_exc()
        return False