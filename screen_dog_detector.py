import os
import sys
import time
import argparse
import cv2
import numpy as np
import mss
import torch
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
    parser = argparse.ArgumentParser(description="Real-Time Windows Screen Capture Dog Detection & Breed Classification")
    parser.add_argument("--monitor", type=int, default=1, help="หมายเลขจอที่ต้องการจับภาพ (default: 1 จอหลัก)")
    parser.add_argument("--scale", type=float, default=0.75, help="อัตราย่อขนาดจอเพื่อความเร็ว FPS (default: 0.75)")
    parser.add_argument("--conf", type=float, default=0.35, help="เกณฑ์ความมั่นใจขั้นต่ำ (default: 0.35)")
    parser.add_argument("--center-crop", action="store_true", help="จับภาพเฉพาะกึ่งกลางจอ 800x600 เพื่อความเร็วสูงสุด")
    args = parser.parse_args()

    print("=" * 70)
    print("🖥️ ระบบตรวจจับและแยกสายพันธุ์สุนัขจากหน้าจอ Windows แบบ Real-Time")
    print("=" * 70)

    dataset_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(dataset_dir, "runs_breed_classification")

    # 1. ตรวจสอบการใช้งาน CUDA GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        device = 0
        print(f"⚡ ตรวจพบ GPU: {gpu_name}")
        print("🚀 เปิดใช้งาน Hardware Acceleration ผ่าน CUDA")
    else:
        device = "cpu"
        print("⚠️ ใช้งานโหมด CPU")

    # 2. โหลดโมเดล
    print("⏳ กำลังโหลดโมเดลตรวจจับตำแหน่ง (YOLOv8 Detection)...")
    detector = YOLO("yolov8n.pt")

    best_weights, run_name = get_latest_best_model(runs_dir)
    if best_weights:
        print(f"🎯 โหลดโมเดลแยกสายพันธุ์จากรอบ: '{run_name}'")
        print(f"📁 Path: {best_weights}")
        classifier = YOLO(best_weights)
    else:
        print("⚠️ ไม่พบโมเดลที่เทรนเอง กำลังใช้ 'yolov8n-cls.pt' แทน")
        classifier = YOLO("yolov8n-cls.pt")

    # 3. เตรียมระบบ Screen Capture
    sct = mss.mss()
    if args.monitor >= len(sct.monitors):
        print(f"❌ ไม่พบจอหมายเลข {args.monitor} จะใช้งานจอหลัก (Monitor 1)")
        mon_idx = 1
    else:
        mon_idx = args.monitor

    monitor = sct.monitors[mon_idx]
    orig_w, orig_h = monitor["width"], monitor["height"]
    print(f"🖥️ จับภาพจอหมายเลข {mon_idx}: ความละเอียด {orig_w}x{orig_h} พิกเซล")

    if args.center_crop:
        crop_w, crop_h = 960, 640
        left = monitor["left"] + (orig_w - crop_w) // 2
        top = monitor["top"] + (orig_h - crop_h) // 2
        capture_area = {"left": left, "top": top, "width": crop_w, "height": crop_h}
        print(f"🎯 โหมดกึ่งกลางจอ: จับภาพพื้นที่ {crop_w}x{crop_h} พิกเซล")
    else:
        capture_area = monitor

    print("\n🚀 เริ่มต้นจับภาพหน้าจอแบบ Real-Time...")
    print("💡 วิธีปิด: คลิกที่หน้าต่างแสดงผลแล้วกดปุ่ม 'q' บนคีย์บอร์ด")
    print("=" * 70)

    colors = [
        (0, 255, 127), (255, 105, 180), (30, 144, 255), (255, 215, 0),
        (255, 69, 0), (147, 112, 219), (0, 255, 255), (50, 205, 50),
        (238, 130, 238), (0, 191, 255)
    ]

    fps_list = []
    prev_time = time.time()

    cv2.namedWindow("Real-Time Windows Screen Dog Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Real-Time Windows Screen Dog Detection", 960, 540)

    try:
        while True:
            # 1. จับภาพหน้าจอ
            screenshot = sct.grab(capture_area)
            # แปลงภาพจาก BGRA (mss) เป็น BGR (OpenCV)
            frame = np.array(screenshot)[:, :, :3]

            # ย่อขนาดลงเล็กน้อยเพื่อเพิ่ม FPS (ถ้าตั้งค่า scale ไว้)
            if args.scale != 1.0 and not args.center_crop:
                new_w = int(frame.shape[1] * args.scale)
                new_h = int(frame.shape[0] * args.scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            h, w, _ = frame.shape

            # 2. ตรวจจับตำแหน่งสุนัข (COCO class 16 = 'dog')
            det_results = detector.predict(frame, classes=[16], conf=args.conf, verbose=False, device=device)[0]

            # 3. วนลูปตีกรอบและระบุสายพันธุ์
            for box in det_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                cropped_dog = frame[y1:y2, x1:x2]
                if cropped_dog.size == 0:
                    continue

                # ส่งภาพครอปไปทายสายพันธุ์
                cls_res = classifier.predict(cropped_dog, verbose=False, device=device)[0]
                top1_id = cls_res.probs.top1
                breed_name = cls_res.names[top1_id]
                precision = cls_res.probs.top1conf.item()

                color = colors[top1_id % len(colors)]

                # วาดกรอบสี่เหลี่ยม
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                # ป้ายชื่อสายพันธุ์ + ค่า Precision
                label = f"{breed_name}: {precision:.1%}"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                tag_y1 = max(0, y1 - text_h - 12)
                cv2.rectangle(frame, (x1, tag_y1), (x1 + text_w + 12, y1), color, -1)
                cv2.putText(
                    frame,
                    label,
                    (x1 + 6, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA
                )

            # คำนวณและแสดง FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time
            fps_list.append(fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
            avg_fps = sum(fps_list) / len(fps_list)

            # แสดงแถบ FPS ที่มุมบนซ้าย
            fps_text = f"FPS: {avg_fps:.1f} | Dogs: {len(det_results.boxes)}"
            cv2.rectangle(frame, (10, 10), (240, 50), (0, 0, 0), -1)
            cv2.putText(frame, fps_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # แสดงผลภาพ
            cv2.imshow("Real-Time Windows Screen Dog Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n⏹️ สั่งหยุดการทำงาน")
    finally:
        cv2.destroyAllWindows()
        print("✅ ปิดโปรแกรมและคืนทรัพยากรเรียบร้อยครับ")

if __name__ == "__main__":
    main()
