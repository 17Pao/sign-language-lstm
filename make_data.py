import cv2
import mediapipe as mp
import numpy as np
import csv
import os
from collections import deque

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

SEQUENCE_LENGTH = 30  # Số khung hình cần ghi lại
data_sequence = deque(maxlen=SEQUENCE_LENGTH)  # Lưu chuỗi tọa độ bàn tay

# Mở webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Dictionary lưu file writers
file_writers = {}


def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.append(lm.x)
                landmark_list.append(lm.y)

            data_sequence.append(landmark_list)

            if len(data_sequence) == SEQUENCE_LENGTH:
                return np.array(data_sequence)
    else:
        # Nếu không phát hiện tay, thêm dữ liệu rỗng
        data_sequence.append([0.0] * 42)  # 21 điểm * 2 (x,y)

    return None


def check_data_validity(landmarks):
    """Kiểm tra tính hợp lệ của dữ liệu"""
    # Kiểm tra xem có đủ 30 frames không
    if len(landmarks) != SEQUENCE_LENGTH:
        return False

    # Kiểm tra xem có frame nào bị thiếu dữ liệu không
    for frame in landmarks:
        if len(frame) != 42:  # 21 điểm * 2 (x,y)
            return False

    return True


def update_samples_count():
    """Cập nhật số mẫu đã thu thập cho mỗi chữ cái"""
    for label in samples_collected:
        csv_filename = f"hand_gesture_{label}.csv"
        if os.path.exists(csv_filename):
            with open(csv_filename, 'r') as f:
                samples_collected[label] = sum(1 for line in f)


def print_samples_count():
    """In ra số lượng mẫu đã thu thập cho mỗi chữ cái (A-Z)"""
    for i in range(26):
        label = chr(65 + i)
        csv_filename = f"hand_gesture_{label}.csv"
        if os.path.exists(csv_filename):
            with open(csv_filename, 'r') as f:
                count = sum(1 for line in f)
            print(f"{label}: {count} mẫu")
        else:
            print(f"{label}: 0 mẫu (chưa có file)")


def main():
    global saving, save_label, frame_count, samples_collected

    # Khởi tạo các biến trạng thái
    saving = False
    save_label = None
    frame_count = 0
    samples_collected = {}  # Đếm số mẫu đã thu thập cho mỗi chữ cái

    # Khởi tạo số mẫu đã thu thập
    for i in range(26):
        label = chr(65 + i)
        samples_collected[label] = 0
    update_samples_count()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Lật frame để hiển thị như gương
        frame = cv2.flip(frame, 1)

        landmarks = extract_landmarks(frame)
        if landmarks is not None and saving:
            if check_data_validity(landmarks):
                csv_filename = f"hand_gesture_{save_label}.csv"

                if save_label not in file_writers:
                    file_exists = os.path.exists(csv_filename)
                    file = open(csv_filename, mode='a', newline='')
                    writer = csv.writer(file)
                    file_writers[save_label] = (file, writer)

                    if not file_exists:
                        print(f"Created new file: {csv_filename}")

                _, writer = file_writers[save_label]
                writer.writerow(landmarks.flatten())
                frame_count += 1
                samples_collected[save_label] += 1

                # Dừng tự động sau khi đủ số frames
                if frame_count >= SEQUENCE_LENGTH:
                    saving = False
                    frame_count = 0
                    save_label = None
                    print(f"Sample saved for {save_label}")
            else:
                print("Invalid data detected, skipping...")

        # Hiển thị thông tin
        y_pos = 30
        cv2.putText(frame, "Press A-Z to save gestures", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 30

        if saving:
            cv2.putText(frame, f"Saving: {save_label} ({frame_count}/{SEQUENCE_LENGTH})",
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_pos += 30

        # Hiển thị số mẫu đã thu thập
        cv2.putText(frame, "Samples collected:", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 30

        # Hiển thị 5 chữ cái đầu tiên
        for i in range(5):
            label = chr(65 + i)
            cv2.putText(frame, f"{label}: {samples_collected[label]}",
                        (10 + i * 100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Hiển thị 5 chữ cái tiếp theo
        y_pos += 30
        for i in range(5, 10):
            label = chr(65 + i)
            cv2.putText(frame, f"{label}: {samples_collected[label]}",
                        (10 + (i - 5) * 100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        y_pos += 30
        for i in range(10, 15):
            label = chr(65 + i)
            cv2.putText(frame, f"{label}: {samples_collected[label]}",
                        (10 + (i - 10) * 100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        y_pos += 30
        for i in range(15, 20):
            label = chr(65 + i)
            cv2.putText(frame, f"{label}: {samples_collected[label]}",
                        (10 + (i - 15) * 100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        y_pos += 30
        for i in range(20, 26):
            label = chr(65 + i)
            cv2.putText(frame, f"{label}: {samples_collected[label]}",
                        (10 + (i - 20) * 100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow("Hand Tracking", frame)

        key = cv2.waitKey(1) & 0xFF

        if ord('a') <= key <= ord('z'):  # Nhấn phím từ A-Z để bắt đầu lưu
            save_label = chr(key).upper()
            saving = True
            frame_count = 0
            print(f"Started saving: {save_label}")
        elif key == 27:  # ESC để thoát
            break

    # Đóng tất cả file CSV đã mở
    for file, _ in file_writers.values():
        file.close()

    print("\nData collection completed!")
    print("\nSamples collected for each letter:")
    for label in samples_collected:
        print(f"{label}: {samples_collected[label]}")


if __name__ == "__main__":
    print_samples_count()
    main()
    cap.release()
    cv2.destroyAllWindows()