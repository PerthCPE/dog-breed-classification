import os
import sys
import argparse
import datetime
import pandas as pd
import torch
from ultralytics import YOLO

def scan_runs(runs_path):
    """สแกนหารอบการเทรนทั้งหมดที่บันทึกไว้"""
    history = []
    if not os.path.exists(runs_path):
        return history

    for item in os.listdir(runs_path):
        folder = os.path.join(runs_path, item)
        if not os.path.isdir(folder):
            continue
        best_pt = os.path.join(folder, "weights", "best.pt")
        last_pt = os.path.join(folder, "weights", "last.pt")
        csv_file = os.path.join(folder, "results.csv")

        info = {
            "name": item,
            "folder": folder,
            "best": best_pt if os.path.exists(best_pt) else None,
            "last": last_pt if os.path.exists(last_pt) else None,
            "epochs": "-",
            "top1": "-",
            "top5": "-",
            "mtime": os.path.getmtime(folder)
        }
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                df.columns = [c.strip() for c in df.columns]
                last_row = df.iloc[-1]
                info["epochs"] = int(last_row.get("epoch", len(df)))
                if "metrics/accuracy_top1" in df.columns:
                    info["top1"] = f"{last_row['metrics/accuracy_top1']:.2%}"
                if "metrics/accuracy_top5" in df.columns:
                    info["top5"] = f"{last_row['metrics/accuracy_top5']:.2%}"
            except Exception:
                pass
        history.append(info)
    history.sort(key=lambda x: x["mtime"], reverse=True)
    return history

def main():
    parser = argparse.ArgumentParser(description="Dog Breed Training Script with Multi-Run Versioning")
    parser.add_argument("--name", type=str, default=None, help="ชื่อรอบการเทรน (ค่าเริ่มต้น: auto-timestamp)")
    parser.add_argument("--epochs", type=int, default=15, help="จำนวนรอบที่ต้องการเทรน (default: 15)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--imgsz", type=int, default=224, help="ขนาดรูปภาพ (default: 224)")
    parser.add_argument("--resume", action="store_true", help="เทรนต่อจาก checkpoint ล่าสุด (last.pt)")
    parser.add_argument("--list", action="store_true", help="ดูประวัติการเทรนทั้งหมด")
    args = parser.parse_args()

    dataset_path = os.path.dirname(os.path.abspath(__file__))
    runs_path = os.path.join(dataset_path, "runs_breed_classification")
    os.makedirs(runs_path, exist_ok=True)

    # โหมดดูประวัติการเทรน
    if args.list:
        runs = scan_runs(runs_path)
        print("=" * 70)
        print("📊 ประวัติการเทรนทั้งหมด (Training History):")
        print("=" * 70)
        if not runs:
            print("ยังไม่มีรอบการเทรนที่บันทึกไว้")
        else:
            table = []
            for r in runs:
                status = "✅ มี best.pt" if r["best"] else "❌ ไม่มี"
                table.append({
                    "Run Name": r["name"],
                    "Epochs": r["epochs"],
                    "Top-1 Acc": r["top1"],
                    "Top-5 Acc": r["top5"],
                    "สถานะ": status
                })
            print(pd.DataFrame(table).to_string(index=False))
        print("=" * 70)
        return

    print("=" * 65)
    print("🐶 ระบบ Train โมเดลแยกสายพันธุ์สุนัข (บันทึกแยกรอบอัตโนมัติ)")
    print("=" * 65)

    # ตรวจสอบการใช้งาน CUDA GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        device = 0
        print(f"⚡ ตรวจพบ GPU: {gpu_name}")
        print(f"🚀 ใช้งานโหมดการเทรนผ่าน: CUDA (device={device})")
    else:
        device = "cpu"
        print("⚠️ ใช้งานโหมด CPU")

    # กำหนดชื่อโฟลเดอร์สำหรับรอบนี้ (ป้องกันการเขียนทับ)
    if args.name:
        run_name = args.name
    else:
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{now_str}"

    print(f"📁 บันทึกผลลัพธ์รอบนี้ไว้ที่: runs_breed_classification/{run_name}")

    if args.resume:
        runs = scan_runs(runs_path)
        last_weight = None
        for r in runs:
            if r["last"]:
                last_weight = r["last"]
                break
        if not last_weight:
            print("❌ ไม่พบ checkpoint (last.pt) สำหรับเทรนต่อ จะเริ่มเทรนรอบใหม่")
            model = YOLO("yolov8n-cls.pt")
            model.train(
                data=dataset_path,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=device,
                workers=2,
                project=runs_path,
                name=run_name,
                exist_ok=False
            )
        else:
            print(f"🔄 กำลังเทรนต่อจาก Checkpoint: {last_weight}")
            model = YOLO(last_weight)
            model.train(resume=True)
    else:
        print(f"⏳ เริ่มต้นการเทรน {args.epochs} epochs...")
        model = YOLO("yolov8n-cls.pt")
        model.train(
            data=dataset_path,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            workers=2,
            project=runs_path,
            name=run_name,
            exist_ok=False  # ป้องกันการเขียนทับรอบเก่า!
        )

    best_file = os.path.join(runs_path, run_name, "weights", "best.pt")
    print("\n" + "=" * 65)
    print("🎉 การเทรนรอบนี้เสร็จสมบูรณ์!")
    print(f"💾 ไฟล์โมเดลที่ดีที่สุดของรอบนี้: {best_file}")
    print("=" * 65)

if __name__ == "__main__":
    main()
