from db import get_connection

def add_employee(employee_code, full_name, image_dir, encoding_file):
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO employees (employee_code, full_name, image_dir, encoding_file)
                    VALUES (%s, %s, %s, %s)
                """, (employee_code, full_name, image_dir, encoding_file))
            conn.commit()
        print("เพิ่มพนักงานสำเร็จ")
    except Exception as e:
        print("เกิดข้อผิดพลาด:", e)

if __name__ == "__main__":
    add_employee("EMP003", "Charlie", "known_faces/Charlie", "encodings/charlie.pkl")