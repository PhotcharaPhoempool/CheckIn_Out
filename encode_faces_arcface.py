"""
encode_faces_arcface.py — สร้าง encodings.pkl ด้วย ArcFace (512d)
===================================================================
รองรับ 2 แหล่งรูปภาพ:

  1. โฟลเดอร์:
     dataset/beam/img1.jpg, dataset/somchai/img1.jpg

  2. URL (JSON):
     urls.json → [{"face_label":"beam","urls":["https://...jpg"]}]

วิธีใช้:
  python encode_faces_arcface.py                              ← โฟลเดอร์ dataset/
  python encode_faces_arcface.py --dir my_faces               ← โฟลเดอร์อื่น
  python encode_faces_arcface.py --urls urls.json             ← จาก URL
  python encode_faces_arcface.py --dir dataset --urls urls.json  ← ทั้งสอง
"""

import os
import cv2
import json
import pickle
import argparse
import numpy as np
import urllib.request
from insightface.app import FaceAnalysis


def _load_insightface():
    print("กำลังโหลด InsightFace (ArcFace)...")
    import onnxruntime as ort
    app = FaceAnalysis(
        name="buffalo_l",
        providers=ort.get_available_providers(),
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("โหลดสำเร็จ!\n")
    return app


def _download_image(url: str):
    """ดาวน์โหลดรูปจาก URL → numpy BGR array"""
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (FaceEncoder/1.0)"
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"    ดาวน์โหลดไม่สำเร็จ: {e}")
        return None


def _encode_face(app, img, source: str):
    """ตรวจจับหน้า → embedding 512d หรือ None"""
    if img is None:
        print(f"    อ่านไม่ได้: {source}")
        return None
    faces = app.get(img)
    if not faces:
        print(f"    ไม่พบหน้า: {source}")
        return None
    return faces[0].embedding


# ─── แหล่งที่ 1: โฟลเดอร์ ────────────────────

def encode_from_folder(app, dataset_dir: str):
    encodings, names, skipped = [], [], 0
    persons = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])
    if not persons:
        print(f"  ไม่พบโฟลเดอร์คนใน {dataset_dir}/")
        return encodings, names, skipped

    print(f"[FOLDER] พบ {len(persons)} คน ใน {dataset_dir}/")
    for person in persons:
        person_dir = os.path.join(dataset_dir, person)
        images = [f for f in os.listdir(person_dir)
                  if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp"))]
        if not images:
            print(f"  {person}: ไม่มีรูป — ข้าม")
            continue
        count = 0
        for img_name in images:
            path = os.path.join(person_dir, img_name)
            emb = _encode_face(app, cv2.imread(path), path)
            if emb is not None:
                encodings.append(emb)
                names.append(person)
                count += 1
            else:
                skipped += 1
        print(f"  {person}: {count}/{len(images)} รูป")
    return encodings, names, skipped


# ─── แหล่งที่ 2: URL (JSON) ──────────────────

def encode_from_urls(app, urls_file: str):
    """
    JSON format:
    [
      {"face_label": "beam",    "urls": ["https://example.com/beam1.jpg"]},
      {"face_label": "somchai", "urls": ["https://example.com/s1.jpg", "https://example.com/s2.jpg"]}
    ]
    """
    encodings, names, skipped = [], [], 0
    with open(urls_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[URL] พบ {len(data)} คน ใน {urls_file}")
    for entry in data:
        label = entry.get("face_label", "")
        urls  = entry.get("urls", [])
        if not label or not urls:
            continue
        count = 0
        for url in urls:
            print(f"  {label}: ดาวน์โหลด {url[:60]}...")
            img = _download_image(url)
            emb = _encode_face(app, img, url)
            if emb is not None:
                encodings.append(emb)
                names.append(label)
                count += 1
            else:
                skipped += 1
        print(f"  {label}: {count}/{len(urls)} รูป")
    return encodings, names, skipped


# ─── Main ────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Encode faces (folder + URL)")
    parser.add_argument("--dir",  default=None, help="โฟลเดอร์ dataset")
    parser.add_argument("--urls", default=None, help="JSON ไฟล์ URL รูปภาพ")
    parser.add_argument("--out",  default="encodings.pkl", help="output file")
    args = parser.parse_args()

    if args.dir is None and args.urls is None:
        args.dir = "dataset"

    app = _load_insightface()
    all_enc, all_names, total_skip = [], [], 0

    if args.dir and os.path.isdir(args.dir):
        enc, nam, skip = encode_from_folder(app, args.dir)
        all_enc.extend(enc); all_names.extend(nam); total_skip += skip
    elif args.dir:
        print(f"ไม่พบโฟลเดอร์: {args.dir}/")

    if args.urls and os.path.isfile(args.urls):
        enc, nam, skip = encode_from_urls(app, args.urls)
        all_enc.extend(enc); all_names.extend(nam); total_skip += skip
    elif args.urls:
        print(f"ไม่พบไฟล์: {args.urls}")

    if not all_enc:
        print("\nไม่มี encoding เลย!")
        return

    with open(args.out, "wb") as f:
        pickle.dump({
            "encodings": all_enc, "names": all_names,
            "model": "arcface_buffalo_l", "dimensions": 512,
        }, f)

    unique = sorted(set(all_names))
    print(f"\n{'='*50}")
    print(f"บันทึกสำเร็จ: {args.out}")
    print(f"  คน:        {len(unique)}  ({', '.join(unique)})")
    print(f"  Encodings: {len(all_enc)}")
    print(f"  ข้าม:       {total_skip}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()