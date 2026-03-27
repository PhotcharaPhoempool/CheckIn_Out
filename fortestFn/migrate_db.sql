-- =====================================================
-- migrate_db.sql — ปรับโครงสร้างฐานข้อมูลใหม่
-- =====================================================
-- เปลี่ยนจาก:
--   employees.id (auto increment) เป็น primary key
-- เป็น:
--   employees.employee_code (เลขบัตรประจำตัว) เป็น primary key
--   แยก title (คำนำหน้า) + first_name + last_name
--
-- วิธีใช้:
--   mysql -u root -p face_attendance < migrate_db.sql
--   หรือ psql -U postgres -d face_attendance -f migrate_db.sql
-- =====================================================

-- ===== สำรองตารางเดิม (ถ้ามี) =====
-- ALTER TABLE employees RENAME TO employees_old;
-- ALTER TABLE attendance_logs RENAME TO attendance_logs_old;

-- ===== ตาราง employees ใหม่ =====
CREATE TABLE IF NOT EXISTS employees (
    employee_code   VARCHAR(20)  PRIMARY KEY,          -- เลขบัตรประจำตัว (ไม่ซ้ำกัน)
    title           VARCHAR(20)  NOT NULL DEFAULT '',  -- คำนำหน้า: นาย, นาง, นางสาว
    first_name      VARCHAR(100) NOT NULL,             -- ชื่อ
    last_name       VARCHAR(100) NOT NULL DEFAULT '',   -- นามสกุล
    face_label      VARCHAR(100) NOT NULL,             -- ชื่อโฟลเดอร์ใน dataset (ใช้จับคู่กับ face recognition)
    department      VARCHAR(100) DEFAULT '',           -- แผนก (optional)
    is_active       BOOLEAN      DEFAULT TRUE,         -- ยังทำงานอยู่ไหม
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
);

-- Index สำหรับค้นหาด้วย face_label (ใช้บ่อยสุดตอน face recognition)
CREATE INDEX IF NOT EXISTS idx_employees_face_label ON employees(face_label);

-- ===== ตาราง attendance_logs ใหม่ =====
CREATE TABLE IF NOT EXISTS attendance_logs (
    id              SERIAL       PRIMARY KEY,          -- auto increment log ID
    employee_code   VARCHAR(20)  NOT NULL,             -- เลขบัตร (FK → employees)
    status          VARCHAR(10)  NOT NULL,             -- 'IN' หรือ 'OUT'
    camera_name     VARCHAR(50)  DEFAULT '',           -- ชื่อกล้อง
    check_time      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (employee_code) REFERENCES employees(employee_code)
);

CREATE INDEX IF NOT EXISTS idx_attendance_code_date
    ON attendance_logs(employee_code, check_time);


-- ===== ตัวอย่างข้อมูล =====
-- INSERT INTO employees (employee_code, title, first_name, last_name, face_label, department)
-- VALUES
--     ('EMP-001', 'นาย',    'สมชาย',  'ใจดี',    'somchai',  'IT'),
--     ('EMP-002', 'นางสาว', 'สมหญิง', 'รักดี',   'somying',  'HR'),
--     ('EMP-003', 'นาง',    'วิภา',   'สุขสันต์', 'vipa',     'Finance');
--
-- หมายเหตุ:
--   face_label = ชื่อโฟลเดอร์ใน dataset/ ที่ encode_faces_arcface.py ใช้
--   เช่น dataset/somchai/img1.jpg → face_label = "somchai"
--   ระบบจะ match ชื่อจาก face recognition กับ face_label ในตาราง
