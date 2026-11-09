# -*- coding: utf-8 -*-
"""
evaluate_metrics.py
ระบบวิเคราะห์และแสดงค่าวัดผลโมเดลจำแนกสายพันธุ์สุนัข (In-Depth Model Evaluation)
รองรับทั้งตารางสรุป, Classification Report รายสายพันธุ์, Confusion Matrix และกราฟ Heatmap
"""

import os
import glob
import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix

def run_evaluation(weights_path=None, split="valid", show_plot=True):
    dataset_dir = r"C:\Users\perth\Downloads\My First Project.v1i.folder"
    runs_dir = os.path.join(dataset_dir, "runs_breed_classification")

    # ค้นหาโมเดล best.pt ล่าสุดอัตโนมัติหากไม่ระบุ
    if weights_path is None or not os.path.exists(weights_path):
        candidates = glob.glob(os.path.join(runs_dir, "**", "best.pt"), recursive=True)
        if candidates:
            candidates.sort(key=os.path.getmtime, reverse=True)
            weights_path = candidates[0]
        else:
            weights_path = os.path.join(runs_dir, "dog_10_breeds", "weights", "best.pt")

    if not os.path.exists(weights_path):
        print(f"❌ ไม่พบไฟล์โมเดล: {weights_path}")
        return

    eval_dir = os.path.join(dataset_dir, split)
    if not os.path.exists(eval_dir):
        print(f"❌ ไม่พบโฟลเดอร์ชุดข้อมูล: {eval_dir}")
        return

    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"

    print("=" * 85)
    print("🎯 ระบบวิเคราะห์ค่าวัดผลโมเดลจำแนกสายพันธุ์สุนัข (DOG BREED MODEL EVALUATION)")
    print(f"📁 ไฟล์โมเดลที่ใช้ประเมิน : {weights_path}")
    print(f"📊 ชุดข้อมูลที่ใช้วัดผล    : {eval_dir} ({split.upper()} SET)")
    print(f"⚡ อุปกรณ์ประมวลผล       : {device_name}")
    print("=" * 85)

    model = YOLO(weights_path)
    classes = sorted([d for d in os.listdir(eval_dir) if os.path.isdir(os.path.join(eval_dir, d))])

    y_true = []
    y_pred = []
    y_pred_probs = []
    top5_correct = 0
    total_samples = 0

    print(f"\n⏳ กำลังเริ่มทดสอบและคำนวณค่าวัดผลบนภาพจำนวนทั้งหมดในชุด '{split}'...")
    start_time = time.time()

    for cls_name in classes:
        cls_dir = os.path.join(eval_dir, cls_name)
        img_paths = glob.glob(os.path.join(cls_dir, "*.jpg")) + glob.glob(os.path.join(cls_dir, "*.png"))
        for p in img_paths:
            res = model.predict(p, verbose=False, device=device)[0]
            top1_name = res.names[res.probs.top1]
            top5_names = [res.names[i] for i in res.probs.top5]

            y_true.append(cls_name)
            y_pred.append(top1_name)
            y_pred_probs.append(res.probs.top1conf.item())

            if cls_name in top5_names:
                top5_correct += 1
            total_samples += 1

    eval_time = time.time() - start_time
    fps = total_samples / eval_time if eval_time > 0 else 0

    # ==========================================
    # 1. Classification Report (รายสายพันธุ์)
    # ==========================================
    rep_dict = classification_report(y_true, y_pred, target_names=classes, output_dict=True, digits=4)
    
    table_rows = []
    for cls_name in classes:
        c_data = rep_dict[cls_name]
        table_rows.append({
            "สายพันธุ์ (Dog Breed)": cls_name,
            "Precision": f"{c_data['precision']:.2%}",
            "Recall": f"{c_data['recall']:.2%}",
            "F1-Score": f"{c_data['f1-score']:.2%}",
            "จำนวนรูปจริง (Support)": int(c_data['support'])
        })

    df_report = pd.DataFrame(table_rows)
    print("\n" + "=" * 85)
    print("📋 [1] ตารางค่าวัดผลรายสายพันธุ์ (Per-Class Performance Metrics):")
    print("=" * 85)
    print(df_report.to_string(index=False))

    # ==========================================
    # 2. Overall Summary Metrics
    # ==========================================
    acc = rep_dict["accuracy"]
    macro_p = rep_dict["macro avg"]["precision"]
    macro_r = rep_dict["macro avg"]["recall"]
    macro_f1 = rep_dict["macro avg"]["f1-score"]
    weighted_f1 = rep_dict["weighted avg"]["f1-score"]
    top5_acc = top5_correct / total_samples

    print("\n" + "=" * 85)
    print("🏆 [2] สรุปค่าวัดผลภาพรวมทั้งหมด (Overall Model Performance):")
    print("=" * 85)
    print(f"  ⭐ Top-1 Accuracy   : {acc:.2%}  (โมเดลทายสายพันธุ์อันดับ 1 ถูกต้อง)")
    print(f"  ⭐ Top-5 Accuracy   : {top5_acc:.2%}  (สายพันธุ์จริงติดอยู่ใน Top 5 ที่โมเดลคาดเดา)")
    print(f"  ⭐ Macro Precision  : {macro_p:.2%}  (ความแม่นยำเฉลี่ยแบบให้น้ำหนักทุกสายพันธุ์เท่ากัน)")
    print(f"  ⭐ Macro Recall     : {macro_r:.2%}  (ความครบถ้วนในการตรวจจับเฉลี่ยทุกสายพันธุ์)")
    print(f"  ⭐ Macro F1-Score   : {macro_f1:.2%}  (คะแนนเฉลี่ยฮาร์โมนิกของโมเดล)")
    print(f"  ⭐ Weighted F1      : {weighted_f1:.2%}  (คะแนนเฉลี่ยแบบถ่วงน้ำหนักตามจำนวนรูปจริง)")
    print(f"  ⚡ ความเร็วทดสอบ    : {eval_time:.2f} วินาที ({fps:.1f} ภาพ/วินาที บน {device_name})")
    print("=" * 85)

    # ==========================================
    # 3. Confusion Matrix
    # ==========================================
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    df_cm = pd.DataFrame(cm, index=[c[:10] for c in classes], columns=[c[:10] for c in classes])

    print("\n" + "=" * 85)
    print("🧩 [3] เมทริกซ์ความสับสน (Confusion Matrix Table):")
    print("     (แกนตั้ง = สายพันธุ์จริง | แกนนอน = สายพันธุ์ที่โมเดลทาย)")
    print("=" * 85)
    print(df_cm.to_string())

    # ==========================================
    # 4. Top Misclassifications (วิเคราะห์จุดที่โมเดลสับสนบ่อยสุด)
    # ==========================================
    confusions = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i][j] > 0:
                confusions.append((classes[i], classes[j], cm[i][j]))
    confusions.sort(key=lambda x: x[2], reverse=True)

    print("\n" + "=" * 85)
    print("🔍 [4] จุดที่โมเดลทายสับสนมากที่สุด 5 อันดับแรก (Top Misclassifications):")
    print("=" * 85)
    for true_cls, pred_cls, count in confusions[:5]:
        print(f"  ⚠️  สายพันธุ์จริง: '{true_cls:<18}' ➔ โมเดลทายผิดเป็น: '{pred_cls:<18}' (ผิดไป {count} ภาพ)")
    print("=" * 85)

    # ==========================================
    # 5. Export Heatmap Plot (Confusion Matrix)
    # ==========================================
    save_fig_path = os.path.join(dataset_dir, f"evaluation_confusion_matrix_{split}.png")
    plt.figure(figsize=(10, 8), dpi=150)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f"Dog Breeds Confusion Matrix ({split.upper()} SET)\nTop-1 Acc: {acc:.2%} | Top-5 Acc: {top5_acc:.2%}", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Predicted Breed", fontsize=11, fontweight='bold')
    plt.ylabel("True Breed", fontsize=11, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_fig_path)
    plt.close()

    print(f"\n🖼️ บันทึกรูปกราฟ Confusion Matrix Heatmap เรียบร้อยที่:")
    print(f"   {save_fig_path}")

    return {
        "df_report": df_report,
        "overall": {
            "top1_acc": acc,
            "top5_acc": top5_acc,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f1
        },
        "cm": cm,
        "classes": classes
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dog Breed Model Evaluation Script")
    parser.add_argument("--weights", type=str, default=None, help="Path to weights (best.pt)")
    parser.add_argument("--split", type=str, default="valid", choices=["valid", "test"], help="Dataset split (valid or test)")
    args = parser.parse_args()

    run_evaluation(weights_path=args.weights, split=args.split)
