"""
import_csv.py — นำเข้าข้อมูลพนักงานจาก CSV เข้าฐานข้อมูล
============================================================
CSV format (UTF-16):
  pid,name,url,status,part,sha1
  1111111111111,นาย ปอปี้ ศรีเหลือง,https://...jpg,ok,/path/to/folder,

สิ่งที่ทำ:
  1. อ่าน CSV → แยก คำนำหน้า / ชื่อ / นามสกุล
  2. INSERT เข้าตาราง employees (employee_code = pid)

วิธีใช้:
  python import_csv.py test.csv                  ← import + สร้าง urls.json
  python import_csv.py test.csv --dry-run        ← ดูก่อนว่าจะ import อะไร
  python import_csv.py test.csv --urls-only      ← สร้างแค่ urls.json (ไม่แตะ DB)
"""

import csv
import json
import argparse
import sys

TITLES = ["นางสาว", "นาง", "นาย", "ด.ญ.", "ด.ช.", "Mr.", "Mrs.", "Ms.", "Miss"]


def parse_name(full_name: str) -> tuple:
    """
    แยก "นาย ปอปี้ ศรีเหลือง" → ("นาย", "ปอปี้", "ศรีเหลือง")
    แยก "นางสาว กระหนก กอไก่" → ("นางสาว", "กระหนก", "กอไก่")
    
    ลำดับตรวจ: นางสาว ก่อน นาง (เพราะ "นาง" เป็น prefix ของ "นางสาว")
    """
    name = full_name.strip()
    title = ""

    # เรียงจากยาวไปสั้น เพื่อจับ "นางสาว" ก่อน "นาง"
    for t in sorted(TITLES, key=len, reverse=True):
        if name.startswith(t):
            title = t
            name = name[len(t):].strip()
            break

    parts = name.split()
    if len(parts) >= 2:
        first_name = parts[0]
        last_name = " ".join(parts[1:])
    elif len(parts) == 1:
        first_name = parts[0]
        last_name = ""
    else:
        first_name = full_name.strip()
        last_name = ""

    return title, first_name, last_name


def read_csv(filepath: str) -> list:
    """อ่าน CSV (รองรับ UTF-16 และ UTF-8)"""
    # ลอง UTF-16 ก่อน (BOM \xff\xfe)
    for enc in ["utf-16", "utf-8-sig", "utf-8", "tis-620"]:
        try:
            with open(filepath, "r", encoding=enc) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    print(f"[CSV] อ่านสำเร็จ encoding={enc}  จำนวน {len(rows)} แถว")
                    return rows
        except (UnicodeDecodeError, UnicodeError):
            continue
    print("[ERROR] อ่าน CSV ไม่สำเร็จ — ลอง encoding ทั้งหมดแล้ว")
    return []


def process_rows(rows: list) -> list:
    """แปลง CSV rows → list of employee dicts"""
    employees = []
    for i, row in enumerate(rows):
        pid    = row.get("pid", "").strip()
        name   = row.get("name", "").strip()
        url    = row.get("url", "").strip()
        status = row.get("status", "").strip().lower()

        if not pid or not name:
            print(f"  แถว {i+1}: ข้ามข้อมูลไม่ครบ (pid={pid}, name={name})")
            continue

        title, first_name, last_name = parse_name(name)

        # face_label = pid (ตรงกับชื่อโฟลเดอร์ใน dataset)
        face_label = pid

        # URL ที่ใช้ได้ (status=ok และ url ไม่ใช่ null)
        has_url = (status == "ok"
                   and url
                   and "null" not in url.lower()
                   and url.startswith("http"))

        employees.append({
            "employee_code": pid,
            "title": title,
            "first_name": first_name,
            "last_name": last_name,
            "face_label": face_label,
            "url": url if has_url else None,
            "status": status,
        })

    return employees


def print_preview(employees: list):
    """แสดงตัวอย่างข้อมูลก่อน import"""
    print(f"\n{'='*65}")
    print(f"{'code':<16} {'title':<8} {'first':<12} {'last':<12} {'url':>5}")
    print(f"{'-'*65}")
    for emp in employees:
        url_mark = "OK" if emp["url"] else "SKIP"
        print(f"{emp['employee_code']:<16} {emp['title']:<8} "
              f"{emp['first_name']:<12} {emp['last_name']:<12} {url_mark:>5}")
    print(f"{'='*65}")
    print(f"ทั้งหมด {len(employees)} คน, "
          f"มี URL {sum(1 for e in employees if e['url'])} คน")


def insert_to_db(employees: list):
    """INSERT เข้าตาราง employees"""
    try:
        from db import get_connection
    except ImportError:
        print("[ERROR] ไม่พบ db.py — ไม่สามารถเชื่อมต่อฐานข้อมูล")
        return False

    count = 0
    with get_connection() as conn:
        with conn.cursor() as cur:
            for emp in employees:
                try:
                    cur.execute("""
                        INSERT INTO employees
                            (employee_code, title, first_name, last_name, face_label)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (employee_code) DO UPDATE SET
                            title = EXCLUDED.title,
                            first_name = EXCLUDED.first_name,
                            last_name = EXCLUDED.last_name,
                            face_label = EXCLUDED.face_label
                    """, (
                        emp["employee_code"],
                        emp["title"],
                        emp["first_name"],
                        emp["last_name"],
                        emp["face_label"],
                    ))
                    count += 1
                except Exception as e:
                    print(f"  [ERROR] {emp['employee_code']}: {e}")
        conn.commit()

    print(f"\n[DB] INSERT สำเร็จ {count}/{len(employees)} คน")
    return True


def generate_urls_json(employees: list, output: str = "urls.json"):
    """สร้าง urls.json สำหรับ encode_faces_arcface.py"""
    entries = []
    for emp in employees:
        if emp["url"]:
            entries.append({
                "face_label": emp["face_label"],
                "urls": [emp["url"]],
            })

    if not entries:
        print("[WARN] ไม่มีรายการที่มี URL — ไม่สร้าง urls.json")
        return

    with open(output, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f"[FILE] สร้าง {output} สำเร็จ ({len(entries)} คน)")
    print(f"       ใช้: python encode_faces_arcface.py --urls {output}")


def main():
    parser = argparse.ArgumentParser(description="Import CSV → DB + urls.json")
    parser.add_argument("csv_file", help="ไฟล์ CSV ข้อมูลพนักงาน")
    parser.add_argument("--dry-run", action="store_true",
                        help="แสดงข้อมูลอย่างเดียว ไม่ insert เข้า DB")
    parser.add_argument("--urls-only", action="store_true",
                        help="สร้าง urls.json อย่างเดียว ไม่แตะ DB")
    parser.add_argument("--urls-out", default="urls.json",
                        help="ชื่อไฟล์ urls.json (default: urls.json)")
    args = parser.parse_args()

    # อ่าน CSV
    rows = read_csv(args.csv_file)
    if not rows:
        sys.exit(1)

    # แปลงข้อมูล
    employees = process_rows(rows)
    if not employees:
        print("ไม่มีข้อมูลที่ import ได้")
        sys.exit(1)

    # แสดงตัวอย่าง
    print_preview(employees)

    if args.dry_run:
        print("\n[DRY-RUN] ไม่ได้ import จริง")
        return

    # สร้าง urls.json เสมอ
    generate_urls_json(employees, args.urls_out)

    # INSERT เข้า DB (ถ้าไม่ใช่ --urls-only)
    if not args.urls_only:
        insert_to_db(employees)
    else:
        print("\n[URLS-ONLY] ไม่ได้ insert เข้า DB")


if __name__ == "__main__":
    main()
