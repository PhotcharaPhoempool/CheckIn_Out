"""
encode_faces_arcface.py — สร้าง encodings.pkl ด้วย ArcFace (512d)
===================================================================
โครงสร้าง dataset:
  dataset/
    ├── ชื่อคน1/
    │   ├── img1.jpg
    │   ├── img2.png
    │   └── ...
    ├── ชื่อคน2/
    │   └── ...

วิธีใช้:
  python encode_faces_arcface.py                    ← ใช้โฟลเดอร์ dataset/
  python encode_faces_arcface.py --dir my_faces     ← ใช้โฟลเดอร์ my_faces/
  python encode_faces_arcface.py --out my_db.pkl    ← บันทึกเป็น my_db.pkl

ผลลัพธ์:
  encodings.pkl = {"encodings": [512d array, ...], "names": ["ชื่อ", ...]}
"""

import os
import cv2
import pickle
import argparse
import numpy as np
from insightface.app import FaceAnalysis


def encode_dataset(dataset_dir: str, output_file: str):
    # ─── โหลด InsightFace ───
    print("กำลังโหลด InsightFace (ArcFace)...")
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("โหลดสำเร็จ!\n")

    encodings = []
    names = []
    skipped = 0

    # ─── วนแต่ละคน ───
    persons = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    if not persons:
        print(f"ไม่พบโฟลเดอร์คนใน {dataset_dir}/")
        return

    print(f"พบ {len(persons)} คน:")
    for person_name in persons:
        person_dir = os.path.join(dataset_dir, person_name)
        images = [
            f for f in os.listdir(person_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]

        if not images:
            print(f"  {person_name}: ไม่มีรูป — ข้าม")
            continue

        person_encs = 0
        for img_name in images:
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"    อ่านไม่ได้: {img_path}")
                skipped += 1
                continue

            faces = app.get(img)
            if not faces:
                print(f"    ไม่พบหน้า: {img_name}")
                skipped += 1
                continue

            # ใช้หน้าแรกที่เจอ (ถ้ามีหลายหน้าในรูป)
            embedding = faces[0].embedding
            encodings.append(embedding)
            names.append(person_name)
            person_encs += 1

        print(f"  {person_name}: {person_encs}/{len(images)} รูป")

    # ─── บันทึก ───
    if not encodings:
        print("\nไม่มี encoding เลย!")
        return

    data = {
        "encodings": encodings,
        "names": names,
        "model": "arcface_buffalo_l",
        "dimensions": 512,
    }

    with open(output_file, "wb") as f:
        pickle.dump(data, f)

    print(f"\nบันทึกสำเร็จ: {output_file}")
    print(f"  คนทั้งหมด:    {len(set(names))}")
    print(f"  Encodings:    {len(encodings)}")
    print(f"  ข้ามไป:        {skipped}")
    print(f"  Model:        ArcFace (buffalo_l, 512d)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode faces with ArcFace")
    parser.add_argument("--dir", default="dataset", help="โฟลเดอร์ dataset")
    parser.add_argument("--out", default="encodings.pkl", help="ไฟล์ output")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"ไม่พบโฟลเดอร์: {args.dir}/")
        print("สร้างโฟลเดอร์แบบนี้:")
        print(f"  {args.dir}/")
        print(f"  ├── ชื่อคน1/")
        print(f"  │   ├── img1.jpg")
        print(f"  │   └── img2.jpg")
        print(f"  └── ชื่อคน2/")
        print(f"      └── img1.jpg")
    else:
        encode_dataset(args.dir, args.out)