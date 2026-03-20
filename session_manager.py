"""
session_manager.py — จัดการ session ของพนักงาน
=================================================
- เก็บข้อมูลคนที่เห็น (person_cache)
- เก็บสถานะ liveness (liveness_cache)
- จัดการ check-in / check-out
"""

from datetime import datetime
from attendance_db import get_employee_by_name, mark_attendance
from liveness_engine import LivenessEngine, LivenessState
import config as cfg


class PersonInfo:
    """ข้อมูลของคนหนึ่งคนในเซสชัน"""

    def __init__(self, name: str, now: datetime, snapshot=None):
        self.name        = name
        self.employee_id = None
        self.first_seen  = now
        self.last_seen   = now
        self.snapshot    = snapshot.copy() if snapshot is not None and snapshot.size > 0 else None
        self.checked_in  = False
        self.checked_out = False

    def update_last_seen(self, now: datetime):
        self.last_seen = now


class SessionManager:
    """
    จัดการทุกคนที่กล้องเห็น

    ใช้งาน:
        session = SessionManager()
        person, liveness = session.get_or_create(name, now, face_crop)
        # ... อัปเดต liveness ...
        session.try_checkin(name)
        session.do_checkout(camera_name)
    """

    def __init__(self):
        self.persons: dict[str, PersonInfo]    = {}
        self.liveness: dict[str, LivenessState] = {}
        self.engine = LivenessEngine()

    # ─── สร้าง/ดึงข้อมูลคน ───────────────────

    def get_or_create(self, name: str, now: datetime, face_crop) -> tuple:
        """
        Returns: (PersonInfo, LivenessState)
        ถ้าเจอคนใหม่ → สร้าง PersonInfo + LivenessState
        ถ้ามีแล้ว → อัปเดต last_seen
        """
        now_ts = now.timestamp()

        # สร้าง person
        if name not in self.persons:
            self.persons[name] = PersonInfo(name, now, face_crop)
        else:
            self.persons[name].update_last_seen(now)

        # Retry liveness ถ้า fail แล้วและรอนานพอ
        if name in self.liveness:
            lv = self.liveness[name]
            if lv.failed and not self.persons[name].checked_in:
                elapsed = now_ts - lv.start_ts
                if elapsed > cfg.LIVENESS_TIMEOUT + cfg.LIVENESS_RETRY_AFTER:
                    print(f"[RETRY] {name}")
                    del self.liveness[name]
                    self.persons[name].first_seen = now

        # สร้าง liveness state ถ้ายังไม่มี
        if name not in self.liveness:
            self.liveness[name] = self.engine.create_state(now_ts)

        return self.persons[name], self.liveness[name]

    # ─── Check-in ────────────────────────────

    def try_checkin(self, name: str, camera_name: str) -> bool:
        """ถ้า liveness ผ่านและยังไม่ check-in → บันทึก IN"""
        person = self.persons.get(name)
        lv     = self.liveness.get(name)

        if not person or not lv:
            return False
        if not lv.confirmed or person.checked_in:
            return False

        emp = get_employee_by_name(name)
        if not emp:
            print(f"[WARN] ไม่พบ '{name}' ในฐานข้อมูล")
            return False

        eid = emp[0]
        person.employee_id = eid

        try:
            result = mark_attendance(eid, "IN", camera_name, check_time=person.first_seen)
            person.checked_in = True  # ถือว่าสำเร็จถ้าไม่ raise exception
            print(f"[IN] {name}  eid={eid}  time={person.first_seen.strftime('%H:%M:%S')}")
            return True
        except Exception as e:
            print(f"[IN ERROR] {name}: {e}")
            person.checked_in = True  # set True ไว้เพื่อให้ checkout ทำงานได้
            return False

    # ─── Check-out (เมื่อถึงเวลา) ────────────

    def do_checkout(self, camera_name: str, now: datetime) -> int:
        """บันทึก OUT ทุกคนที่ checked_in แล้ว — return จำนวนที่สำเร็จ"""
        print(f"\n{'='*50}")
        print(f"[CHECKOUT] เวลา {now.strftime('%H:%M:%S')} — เริ่มบันทึก OUT")
        print(f"[CHECKOUT] คนทั้งหมด: {len(self.persons)} คน")

        count = 0
        for name, person in self.persons.items():
            last = person.last_seen or now
            eid  = person.employee_id

            print(f"  [{name}] in={person.checked_in} out={person.checked_out} "
                  f"eid={eid} last={last.strftime('%H:%M:%S')}")

            if not person.checked_in:
                print(f"    → ข้าม: ยังไม่ check-in")
                continue
            if person.checked_out:
                print(f"    → ข้าม: check-out แล้ว")
                continue

            # ลองหา employee_id ถ้ายังไม่มี
            if not eid:
                emp = get_employee_by_name(name)
                if emp:
                    eid = emp[0]
                    person.employee_id = eid
                else:
                    print(f"    → ไม่พบใน DB — ข้าม")
                    continue

            try:
                mark_attendance(eid, "OUT", camera_name, check_time=last)
                person.checked_out = True
                count += 1
                print(f"    → OUT สำเร็จ (last_seen={last.strftime('%H:%M:%S')})")
            except Exception as e:
                print(f"    → ERROR: {e}")

        print(f"[CHECKOUT] สำเร็จ {count} คน")
        print(f"{'='*50}\n")
        return count