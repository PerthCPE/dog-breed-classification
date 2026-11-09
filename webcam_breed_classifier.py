import os
import cv2
from ultralytics import YOLO

def get_latest_best_model(runs_dir):
    """ค้นหาไฟล์ best.pt จากรอบที่เทรนล่าสุด"""
    if not os.path.exists(runs_dir):
        return None, None
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

def main():
    print("=" * 65)
    print("🐶 ระบบตรวจจับสุนัข ตีกรอบ Bounding Box พร้อมระบุสายพันธุ์และ Precision")
    print("=" * 65)

    dataset_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(dataset_dir, "runs_breed_classification")

    # 1. โหลดโมเดลตรวจจับตำแหน่งสุนัข (Detector)
    print("⏳ โหลดโมเดลตรวจจับตำแหน่ง (YOLOv8 Detection)...")
    detector = YOLO("yolov8n.pt")

    # 2. โหลดโมเดลจำแนกสายพันธุ์ล่าสุดอัตโนมัติ
    best_weights, run_name = get_latest_best_model(runs_dir)
    if best_weights:
        print(f"🎯 โหลดโมเดลจำแนกสายพันธุ์จากรอบ: '{run_name}'")
        print(f"📁 Path: {best_weights}")
        classifier = YOLO(best_weights)
    else:
        print("⚠️ ไม่พบโมเดลที่เทรนเอง กำลังใช้ 'yolov8n-cls.pt' แทน")
        classifier = YOLO("yolov8n-cls.pt")

    # 3. เปิดกล้อง Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ไม่สามารถเปิดกล้อง Webcam ได้")
        return

    print("\n🚀 เริ่มต้นตรวจจับผ่านกล้อง Webcam")
    print("💡 วิธีปิด: คลิกที่หน้าต่างแล้วกดปุ่ม 'q' บนคีย์บอร์ด")

    colors = [
        (0, 255, 127), (255, 105, 180), (30, 144, 255), (255, 215, 0),
        (255, 69, 0), (147, 112, 219), (0, 255, 255), (50, 205, 50),
        (238, 130, 238), (0, 191, 255)
    ]

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        h, w, _ = frame.shape

        # ตรวจจับตำแหน่งสุนัข (class 16 = dog)
        det_results = detector.predict(frame, classes=[16], conf=0.35, verbose=False)[0]

        # ตีกรอบและวิเคราะห์สายพันธุ์แต่ละตัว
        for box in det_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            cropped_dog = frame[y1:y2, x1:x2]
            if cropped_dog.size == 0:
                continue

            cls_res = classifier.predict(cropped_dog, verbose=False)[0]
            top1_id = cls_res.probs.top1
            breed_name = cls_res.names[top1_id]
            precision = cls_res.probs.top1conf.item()

            color = colors[top1_id % len(colors)]

            # วาดกรอบสี่เหลี่ยม
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # ป้ายข้อความ
            label = f"{breed_name}: {precision:.1%}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            tag_y1 = max(0, y1 - text_h - 12)
            cv2.rectangle(frame, (x1, tag_y1), (x1 + text_w + 12, y1), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 6, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

        cv2.imshow("Dog Breed Detection (Press 'q' to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ ปิดกล้องเรียบร้อย")

if __name__ == "__main__":
    main()
