import cv2
import mediapipe as mp
from pynput.keyboard import Controller, Key

# ตั้งค่า MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
keyboard = Controller()

# เปิดกล้อง
cap = cv2.VideoCapture(1)

# ตัวแปรสถานะ
prev_hip_y = None  # ค่าตำแหน่งสะโพกจากเฟรมก่อนหน้า
is_walking = False  # สถานะการเดิน
current_turn = None  # สถานะหันซ้าย/ขวา

# ค่ากำหนด
walk_threshold = 0.05  # ความเร็วสะโพกที่ถือว่า "เดิน"
box_width = 120  # ความกว้างของกล่องตรวจจับ
box_height = 100  # ความสูงของกล่องตรวจจับ
box_y_offset = 350  # ย้ายกล่องขึ้นบนจากขอบล่างของเฟรม
box_x_offset = 115  # ปรับระยะชิดเข้ามาใกล้ศูนย์กลางมากขึ้น

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape  # ขนาดของเฟรม
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # กลับภาพจากกล้อง (Flip)
    flipped_frame = cv2.flip(frame, 1)  # 1 คือการกลับแนวนอน

    # กำหนดตำแหน่งของกล่อง L และ R โดยขยับเข้าใกล้ศูนย์กลางมากขึ้น
    left_box = (box_x_offset, height - box_height - box_y_offset, box_x_offset + box_width, height - box_y_offset)
    right_box = (width - box_x_offset - box_width, height - box_height - box_y_offset, width - box_x_offset, height - box_y_offset)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # **การเดิน (`W`)**
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        hip_y = (left_hip.y + right_hip.y) / 2  # ค่ากลางของสะโพก

        if prev_hip_y is not None:
            speed = abs(hip_y - prev_hip_y) / (1 / 35)  # คิดว่า FPS = 35

            if speed > walk_threshold:  # เริ่มเดิน
                if not is_walking:
                    keyboard.press('w')
                    is_walking = True
            else:
                if is_walking:
                    keyboard.release('w')
                    is_walking = False

        prev_hip_y = hip_y  # อัปเดตค่าสะโพก

        # **ตรวจจับศีรษะอยู่ในกล่องไหน (`←` / `→`)**
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        nose_x, nose_y = int(nose.x * width), int(nose.y * height)

        if left_box[0] < nose_x < left_box[2] and left_box[1] < nose_y < left_box[3]:  # อยู่ในกล่อง L
            if current_turn != "left":
                keyboard.press(Key.left)
                keyboard.release(Key.right)
                current_turn = "left"
        elif right_box[0] < nose_x < right_box[2] and right_box[1] < nose_y < right_box[3]:  # อยู่ในกล่อง R
            if current_turn != "right":
                keyboard.press(Key.right)
                keyboard.release(Key.left)
                current_turn = "right"
        else:
            keyboard.release(Key.left)
            keyboard.release(Key.right)
            current_turn = None

    # วาดกล่องตรวจจับ
    cv2.rectangle(flipped_frame, (left_box[0], left_box[1]), (left_box[2], left_box[3]), (0, 0, 255), 2)
    cv2.putText(flipped_frame, 'L', (left_box[0] + 10, left_box[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.rectangle(flipped_frame, (right_box[0], right_box[1]), (right_box[2], right_box[3]), (255, 0, 0), 2)
    cv2.putText(flipped_frame, 'R', (right_box[0] + 10, right_box[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # แสดงภาพที่กลับด้าน
    cv2.imshow('Pose Detection', flipped_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
