import os
import sys
import glob
import cv2
import pandas as pd
from PIL import Image
from ultralytics import YOLO

def get_latest_best_model(runs_dir):
    """ค้นหาไฟล์ best.pt จากรอบที่เทรนล่าสุด"""
    if not os.path.exists(runs_dir):
        return None
    candidates = []
    for item in os.listdir(runs_dir):
        folder = os.path.join(runs_dir, item)
        if not os.path.isdir(folder):
            continue
        best_pt = os.path.join(folder, "weights", "best.pt")
        if os.path.exists(best_pt):
            candidates.append((os.path.getmtime(best_pt), best_pt, item))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1], candidates[0][2]
    return None, None

def predict_dog_breed(image_path=None, model_path=None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(current_dir, "runs_breed_classification")

    # 1. โหลดโมเดล
    if model_path is None:
        best_pt, run_name = get_latest_best_model(runs_dir)
        if best_pt:
            model_path = best_pt
            print(f"🎯 ใช้โมเดลจากรอบ: '{run_name}'")
            print(f"📁 Path: {model_path}")
        else:
            model_path = "yolov8n-cls.pt"
            print("⚠️ ยังไม่พบโมเดลที่เทรนเสร็จ จะใช้ 'yolov8n-cls.pt' แทน")

    model = YOLO(model_path)

    # 2. เลือกภาพตัวอย่างหากไม่ได้ระบุ
    if image_path is None or not os.path.exists(image_path):
        sample_images = glob.glob(os.path.join(current_dir, "test", "*", "*.jpg"))
        if not sample_images:
            print("❌ ไม่พบภาพทดสอบในโฟลเดอร์ test")
            return
        image_path = sample_images[0]

    print(f"\n📸 กำลังวิเคราะห์สายพันธุ์ภาพ: {os.path.basename(image_path)}")
    folder_breed = os.path.basename(os.path.dirname(image_path))
    print(f"🏷️ สายพันธุ์จริงตามโฟลเดอร์ (Ground Truth): {folder_breed}")

    # 3. ทำนายผล
    results = model.predict(source=image_path, verbose=False)
    result = results[0]

    top1_index = result.probs.top1
    top1_name = result.names[top1_index]
    top1_conf = result.probs.top1conf.item()

    print("\n" + "=" * 50)
    print(f"🏆 ผลการทำนายอันดับ 1: {top1_name} (ความมั่นใจ: {top1_conf:.2%})")
    print("=" * 50)

    # แสดง Top 5 ความน่าจะเป็น
    print("📊 5 อันดับสายพันธุ์ที่มีความเป็นไปได้สูงสุด:")
    top5_indices = result.probs.top5
    for rank, idx in enumerate(top5_indices, 1):
        name = result.names[idx]
        prob = result.probs.data[idx].item()
        bar = "█" * int(prob * 30)
        print(f"  {rank}. {name:<20} {prob:>6.2%} {bar}")
    print("=" * 50)

if __name__ == "__main__":
    img_arg = sys.argv[1] if len(sys.argv) > 1 else None
    predict_dog_breed(img_arg)
