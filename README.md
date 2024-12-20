# Hand Gesture to Emoji

โปรเจกต์นี้ใช้ **MediaPipe** และ **OpenCV** เพื่อตรวจจับท่าทางมือจากกล้องเว็บแคม และแสดงคำที่สอดคล้องกับท่าทางที่ตรวจจับได้บนหน้าจอแบบเรียลไทม์


## ความต้องการ

- Python 3.x
- ไลบรารีที่ใช้:
  - `opencv-python`
  - `mediapipe`

คุณสามารถติดตั้งไลบรารีที่จำเป็นได้โดยใช้คำสั่งนี้:

```bash
pip install mediapipe opencv-python
```

## การตั้งค่า

1. **รันสคริปต์**:

```bash
python hand_to_emoji.py
```

- สคริปต์นี้จะเปิดกล้องเว็บแคมของคุณและตรวจจับท่าทางมือ ท่าทางจะถูกแปลงเป็นคำและแสดงผลบนหน้าจอ

2. **ออกจากโปรแกรม**:
   - กดปุ่ม `q` เพื่อออกจากโปรแกรม

## วิธีการใช้งาน

1. รันสคริปต์ Python
2. โปรแกรมจะเปิดกล้องเว็บแคมและตรวจจับท่าทางมือ
3. ท่าทางที่รองรับ
   👍: ยกนิ้วโป้ง (LIKE)
   ✌️: ชูสองนิ้ว (เลข 2)
   🤚: มือเปิด (ท่าทางเริ่มต้นที่ไม่พบท่าทางอื่น)
4. กด `q` เพื่อปิดหน้าต่างและหยุดโปรแกรม

## การแก้ไขปัญหา

- **ไม่สามารถตรวจจับมือได้**: ตรวจสอบให้แน่ใจว่ามือของคุณมองเห็นได้ชัดเจนจากกล้อง
- **กล้องไม่ทำงาน**: หากใช้กล้องภายนอกให้ลองเปลี่ยนการตั้งค่าใน `cv2.VideoCapture(0)` เป็น `cv2.VideoCapture(1)`

## License

โปรเจกต์นี้เป็นโอเพนซอร์ส คุณสามารถนำไปพัฒนาและปรับปรุงเพิ่มเติมได้ตามต้องการ!

---
