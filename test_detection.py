import os
import glob
import cv2
from ultralytics import YOLO

def main():
    print("=" * 50)
    print("🐶 Dog Object Detection Test")
    print("=" * 50)

    # 1. โหลดโมเดล YOLOv8 Nano
    print("⏳ กำลังโหลดโมเดล YOLOv8n...")
    model = YOLO("yolov8n.pt")
    print(f"✅ โหลดโมเดลสำเร็จ! คลาส 16 คือ: '{model.names[16]}'")

    # 2. ค้นหาภาพทดสอบ
    dataset_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(dataset_dir, "test")
    sample_images = glob.glob(os.path.join(test_dir, "*", "*.jpg"))

    if not sample_images:
        print("❌ ไม่พบไฟล์รูปภาพในโฟลเดอร์ test")
        return

    sample_img = sample_images[0]
    print(f"📸 ทดสอบกับรูป: {sample_img}")

    # 3. รัน Object Detection (กรองเฉพาะคลาสสุนัข ID: 16)
    results = model.predict(source=sample_img, classes=[16], conf=0.25)
    result = results[0]

    print(f"📍 ตรวจพบสุนัขทั้งหมด: {len(result.boxes)} ตัว")
    for idx, box in enumerate(result.boxes):
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"   [{idx + 1}] ความมั่นใจ: {conf:.2%} | Bounding Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

    # 4. บันทึกผลลัพธ์ภาพที่ตีกรอบ
    output_dir = os.path.join(dataset_dir, "detection_results")
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "result_" + os.path.basename(sample_img))
    
    annotated_img = result.plot()
    cv2.imwrite(out_file, annotated_img)
    print(f"💾 บันทึกภาพพร้อมตีกรอบแล้วที่: {out_file}")
    print("=" * 50)

if __name__ == "__main__":
    main()
