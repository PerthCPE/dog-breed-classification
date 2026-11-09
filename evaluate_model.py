import os
import glob
import time
import argparse
import pandas as pd
import numpy as np
import torch
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix

def main():
    parser = argparse.ArgumentParser(description="Detailed Model Evaluation & Performance Breakdown")
    parser.add_argument("--weights", type=str, default=None, help="Path to weights file")
    parser.add_argument("--split", type=str, default="valid", choices=["valid", "test"], help="Dataset split to evaluate on (valid or test)")
    args = parser.parse_args()

    dataset_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(dataset_dir, "runs_breed_classification")

    if args.weights and os.path.exists(args.weights):
        weights_path = args.weights
    else:
        weights_path = os.path.join(runs_dir, "dog_10_breeds", "weights", "best.pt")

    if not os.path.exists(weights_path):
        print(f"❌ ไม่พบไฟล์โมเดลที่: {weights_path}")
        return

    eval_dir = os.path.join(dataset_dir, args.split)
    if not os.path.exists(eval_dir):
        print(f"❌ ไม่พบโฟลเดอร์ {args.split}")
        return

    print("=" * 85)
    print(f"🔬 รายงานผลการประเมินโมเดลแบบละเอียด (In-Depth Model Evaluation Report)")
    print(f"📁 โมเดลที่ใช้ประเมิน: {weights_path}")
    print(f"📂 ชุดข้อมูลที่ทดสอบ : {eval_dir} ({args.split.upper()} SET)")
    print("=" * 85)

    device = 0 if torch.cuda.is_available() else "cpu"
    if device == 0:
        print(f"⚡ รันการประเมินผ่าน GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ รันการประเมินผ่าน: CPU")

    model = YOLO(weights_path)
    classes = sorted([d for d in os.listdir(eval_dir) if os.path.isdir(os.path.join(eval_dir, d))])

    y_true = []
    y_pred = []
    top5_correct = 0
    total_samples = 0

    print("\n⏳ กำลังประมวลผลรูปภาพทั้งหมด...")
    start_time = time.time()

    for cls_name in classes:
        img_paths = glob.glob(os.path.join(eval_dir, cls_name, "*.jpg"))
        for p in img_paths:
            res = model.predict(p, verbose=False, device=device)[0]
            top1_name = res.names[res.probs.top1]
            top5_names = [res.names[i] for i in res.probs.top5]

            y_true.append(cls_name)
            y_pred.append(top1_name)

            if cls_name in top5_names:
                top5_correct += 1
            total_samples += 1

    eval_time = time.time() - start_time
    fps = total_samples / eval_time

    # 1. Classification Report แบบตารางละเอียด
    rep_dict = classification_report(y_true, y_pred, target_names=classes, output_dict=True, digits=4)
    
    rows = []
    for cls_name in classes:
        c_data = rep_dict[cls_name]
        rows.append({
            "สายพันธุ์ (Dog Breed)": cls_name,
            "Precision": f"{c_data['precision']:.2%}",
            "Recall": f"{c_data['recall']:.2%}",
            "F1-Score": f"{c_data['f1-score']:.2%}",
            "จำนวนรูปภาพ (Support)": int(c_data['support'])
        })

    df_report = pd.DataFrame(rows)
    print("\n" + "=" * 85)
    print("📋 [1] ตารางค่าวัดผลแยกรายสายพันธุ์ (Per-Class Performance Metrics):")
    print("=" * 85)
    print(df_report.to_string(index=False))

    # สรุปภาพรวม
    acc = rep_dict["accuracy"]
    macro_f1 = rep_dict["macro avg"]["f1-score"]
    top5_acc = top5_correct / total_samples

    print("\n" + "-" * 85)
    print(f"📊 [สรุปภาพรวมทั้งหมด - Overall Performance]")
    print(f"  • Top-1 Accuracy : {acc:.2%}  (ทายสายพันธุ์อันดับ 1 ถูกต้อง)")
    print(f"  • Top-5 Accuracy : {top5_acc:.2%}  (สายพันธุ์จริงติด 1 ใน 5 อันดับแรก)")
    print(f"  • Macro F1-Score : {macro_f1:.2%}  (ความแม่นยำเฉลี่ยถ่วงน้ำหนักเท่ากันทุกคลาส)")
    print(f"  • ความเร็วในการรัน: {eval_time:.2f} วินาที ({fps:.1f} FPS บน {total_samples} ภาพ)")
    print("-" * 85)

    # 2. Confusion Matrix Table
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    df_cm = pd.DataFrame(cm, index=[c[:10] for c in classes], columns=[c[:10] for c in classes])

    print("\n" + "=" * 85)
    print("🔍 [2] ตารางเมทริกซ์ความสับสน (Confusion Matrix Table):")
    print("    (แถวแนวนอน = สายพันธุ์จริง | คอลัมน์แนวตั้ง = สิ่งที่โมเดลทาย)")
    print("=" * 85)
    print(df_cm.to_string())

    # 3. Top Confusion Pairs (วิเคราะห์คู่สายพันธุ์ที่โมเดลทายสลับกันบ่อยที่สุด)
    confusions = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i][j] > 0:
                confusions.append((classes[i], classes[j], cm[i][j]))
    confusions.sort(key=lambda x: x[2], reverse=True)

    print("\n" + "=" * 85)
    print("💡 [3] วิเคราะห์จุดผิดพลาด: คู่สายพันธุ์ที่โมเดลทายสลับกันบ่อยที่สุด (Top Misclassifications):")
    print("=" * 85)
    for true_cls, pred_cls, count in confusions[:5]:
        print(f"  ⚠️  สายพันธุ์จริง: '{true_cls}' ➔ โมเดลทายผิดเป็น: '{pred_cls}' (พบ {count} รูป)")

    print("=" * 85)

if __name__ == "__main__":
    main()
