# 🐕 Dog Breed Classification & Real-Time Detection (10 Breeds)

ระบบจำแนกสายพันธุ์สุนัข 10 สายพันธุ์และตรวจจับตีกรอบแบบ Real-Time ด้วยโมเดล **YOLOv8** พัฒนาและเทรนด้วย GPU CUDA (NVIDIA GeForce RTX 4050)

---

## 📊 ข้อมูลชุดข้อมูลและสายพันธุ์ (Dataset & Classes)
ชุดข้อมูลสุนัขแบ่งเป็น 10 คลาสในโครงสร้าง Folder Classification (`train/`, `valid/`, `test/`):
1. Beagle
2. Boxer
3. Chihuahua
4. Cocker_Spaniel
5. German_Shorthaired
6. Great_Pyrenees
7. Pomeranian
8. Pug
9. Saint_Bernard
10. Samoyed

---

## 🏆 ผลการวัดผลโมเดล (Performance Metrics on Valid Set)
ประเมินผลด้วย `evaluate_metrics.py` บนชุด Validation (498 ภาพ):
* **Top-1 Accuracy:** `83.73%` (ทายสายพันธุ์อันดับ 1 ถูกต้อง)
* **Top-5 Accuracy:** `99.00%` (สายพันธุ์จริงติด 1 ใน 5 อันดับแรก)
* **Macro F1-Score:** `84.09%`
* **Inference Speed:** `~66.7 FPS` บน GPU RTX 4050

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)
```text
├── data.yaml                     # ไฟล์ Config สายพันธุ์สำหรับ YOLO
├── dog_breed_classification.ipynb # สมุดบันทึกหลัก: โหลดโมเดล, ประเมินผล, ทดสอบรูป, เว็บแคม
├── dog_object_detection.ipynb    # สมุดบันทึกตรวจจับสุนัขทั่วไป
├── train_dog_breeds.py           # สคริปต์เทรนโมเดล พร้อมระบบแยกเวอร์ชัน run
├── predict_breed.py              # ทดสอบทำนายภาพเดี่ยว + แสดงผล Top-5
├── evaluate_metrics.py           # วัดผลแบบเจาะลึก (Precision, Recall, F1, Confusion Matrix)
├── webcam_breed_classifier.py    # ตีกรอบตรวจจับและบอกสายพันธุ์สดผ่าน Webcam
├── screen_dog_detector.py        # ตรวจจับน้องหมาบนหน้าจอ Windows แบบ Real-Time
└── runs_breed_classification/    # โฟลเดอร์เก็บน้ำหนักโมเดล (best.pt) และประวัติการเทรน
```

---

## 🚀 วิธีการติดตั้งและใช้งาน (Getting Started)

### 1. ติดตั้ง Dependencies
```powershell
pip install ultralytics opencv-python torch torchvision mss seaborn scikit-learn
```

### 2. วัดผลโมเดล (Evaluation)
```powershell
python evaluate_metrics.py --split valid
```

### 3. เปิดกล้อง Webcam ตีกรอบระบุสายพันธุ์ Real-Time
```powershell
python webcam_breed_classifier.py
```

### 4. จับภาพหน้าจอ Windows ตรวจจับสายพันธุ์สุนัข Real-Time
```powershell
python screen_dog_detector.py
```
