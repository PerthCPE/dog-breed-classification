import cv2
from ultralytics import YOLO

def main():
    print("=" * 50)
    print("🎥 Live Dog Detection via Webcam")
    print("=" * 50)

    # 1. โหลดโมเดล YOLOv8n
    print("⏳ กำลังโหลดโมเดล...")
    model = YOLO("yolov8n.pt")
    print("✅ โหลดโมเดลสำเร็จ! (คลาส 16 = dog)")

    # 2. เปิดกล้อง Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ไม่สามารถเปิดกล้อง Webcam ได้ กรุณาตรวจสอบการเชื่อมต่อหรือการอนุญาตเข้าถึงกล้อง")
        return

    print("🚀 เริ่มตรวจจับสุนัขแบบ Real-Time")
    print("💡 วิธีปิด: คลิกที่หน้าต่างแสดงภาพแล้วกดปุ่ม 'q'")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("❌ ไม่สามารถอ่านภาพจากกล้องได้")
            break

        # ตรวจจับเฉพาะหมา (classes=[16]) ด้วย threshold conf 0.35
        results = model.predict(frame, classes=[16], conf=0.35, verbose=False)
        annotated_frame = results[0].plot()

        cv2.imshow("Dog Object Detection - Real-Time (Press 'q' to quit)", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ ปิดกล้องและโปรแกรมเรียบร้อยแล้ว")

if __name__ == "__main__":
    main()
