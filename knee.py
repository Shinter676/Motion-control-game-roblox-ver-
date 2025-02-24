import cv2
import mediapipe as mp
from pynput.keyboard import Controller, Key

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()
keyboard = Controller()

cap = cv2.VideoCapture(0)

is_walking = False  # สถานะการเดิน
current_turn = None  # สถานะหันซ้าย/ขวา
prev_knee_y = None  # เก็บค่าก่อนหน้าของเข่า

walk_threshold = 0.02  # ค่าความเร็วขั้นต่ำที่จะถือว่าเดิน
box_width = 120  
box_height = 100  
box_y_offset = 350  
box_x_offset = 115  

fps = 50  # สมมติว่ากล้องจับได้ 35 FPS

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # กลับภาพจากกล้อง
    flipped_frame = cv2.flip(frame, 1)

    # กำหนดตำแหน่งของกล่อง L และ R
    left_box = (box_x_offset, height - box_height - box_y_offset, box_x_offset + box_width, height - box_y_offset)
    right_box = (width - box_x_offset - box_width, height - box_height - box_y_offset, width - box_x_offset, height - box_y_offset)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # **การเดิน (W) ใช้เข่าแทนสะโพก**
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        knee_y = (left_knee.y + right_knee.y) / 2  # คำนวณค่ากลางของเข่า

        if prev_knee_y is not None:
            speed = abs(knee_y - prev_knee_y) / (1 / fps)  # คำนวณความเร็ว

            if speed > walk_threshold:  # เริ่มเดิน
                if not is_walking:
                    keyboard.press('w')
                    is_walking = True
            else:
                if is_walking:
                    keyboard.release('w')
                    is_walking = False

        prev_knee_y = knee_y  # ใช้ค่าใหม่ของเข่าแทนสะโพก

        # **ตรวจจับศีรษะอยู่ในกล่องไหน (← / →)**
        left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
        right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]

        # คำนวณตำแหน่งเฉลี่ยของดวงตา
        eye_x = int((left_eye.x + right_eye.x) / 2 * width)
        eye_y = int((left_eye.y + right_eye.y) / 2 * height)

        if left_box[0] < eye_x < left_box[2] and left_box[1] < eye_y < left_box[3]:  
            if current_turn != "left":
                keyboard.press(Key.right)
                keyboard.release(Key.left)
                current_turn = "left"
        elif right_box[0] < eye_x < right_box[2] and right_box[1] < eye_y < right_box[3]:  
            if current_turn != "right":
                keyboard.press(Key.left)
                keyboard.release(Key.right)
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
