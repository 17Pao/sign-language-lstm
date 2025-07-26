import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import json
import os
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load mô hình đã train
gesture_interpreter = tf.lite.Interpreter(model_path="lstm_hand_gesture_model.tflite")
gesture_interpreter.allocate_tensors()
gesture_input_details = gesture_interpreter.get_input_details()
gesture_output_details = gesture_interpreter.get_output_details()

# Load model và label_encoder cho autocomplete
with open('label_encoder.pkl', 'rb') as f:
    autocomplete_label_encoder = pickle.load(f)

# Load TFLite model cho autocomplete
autocomplete_interpreter = tf.lite.Interpreter(model_path="lstm_word_autocomplete.tflite")
autocomplete_interpreter.allocate_tensors()
autocomplete_input_details = autocomplete_interpreter.get_input_details()
autocomplete_output_details = autocomplete_interpreter.get_output_details()

# Bảng mã hóa ký tự (phải giống lúc train)
char_dict = {ch: i+1 for i, ch in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}
maxlen = 12# Phải giống lúc train

# Thêm buffer lưu chuỗi gesture
gesture_buffer = []
BUFFER_MAXLEN = 10  # Số gesture tối đa lưu lại để gợi ý từ

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,  # Tăng threshold lên 0.7
    min_tracking_confidence=0.7    # Tăng threshold lên 0.7
)
mp_draw = mp.solutions.drawing_utils

# Số khung hình cần cho mỗi dự đoán
SEQUENCE_LENGTH = 30  # Khớp với model đã train
NUM_LANDMARKS = 21 * 2  # 21 điểm, mỗi điểm có x, y

# Khởi tạo deque để lưu sequence
data_sequence = deque(maxlen=SEQUENCE_LENGTH)

# Load label mapping từ file JSON
def load_label_mapping():
    try:
        with open('label_mapping.json', 'r') as f:
            label_map = json.load(f)
            # Chuyển đổi string keys thành integers
            label_map = {int(k): v for k, v in label_map.items()}
            return label_map
    except FileNotFoundError:
        print("Error: label_mapping.json not found!")
        return None

# Load label mapping
label_map = load_label_mapping()
if label_map is None:
    print("Please run create_label_mapping.py first!")
    exit()

def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    landmark_list = []
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Chuẩn hóa tọa độ landmarks
            for lm in hand_landmarks.landmark:
                landmark_list.append(lm.x)
                landmark_list.append(lm.y)

            # Thêm vào sequence
            data_sequence.append(landmark_list)
    else:
        # Nếu không phát hiện tay, thêm dữ liệu rỗng
        data_sequence.append([0.0] * NUM_LANDMARKS)

    # Chỉ trả về sequence khi đủ độ dài
    if len(data_sequence) == SEQUENCE_LENGTH:
        return np.array(data_sequence)
    return None


def main():
    # Mở webcam
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    cap = cv2.VideoCapture(0)

    # Đặt độ phân giải
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Tăng lên 800
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Tăng lên 600

    # Biến để theo dõi trạng thái hiển thị
    last_prediction = None
    prediction_counter = 0
    prediction_threshold = 10  # Tăng threshold lên 10
    confidence_min = 0.85     # Tăng confidence threshold lên 0.85
    pending_gesture = None    # Biến lưu gesture đang chờ xác nhận
    applied_words = []  # Danh sách các từ đã chọn

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Lật frame để hiển thị như gương
        frame = cv2.flip(frame, 1)

        # Xử lý phím bấm
        key = cv2.waitKey(1) & 0xFF
        # Xử lý phím số 1-5 để chọn suggest top 1-5
        if key in [ord(str(i)) for i in range(1, 6)]:
            idx = key - ord('1')
            buffer_str = ''.join(gesture_buffer).upper()
            if len(buffer_str) >= 1:
                try:
                    topN = 200
                    seq = [char_dict.get(c, 0) for c in list(buffer_str)]
                    seq = pad_sequences([seq], maxlen=maxlen, padding='pre')
                    pred = predict_tflite(seq)
                    idxs = np.argsort(pred[0])[-topN:][::-1]
                    all_suggestions = autocomplete_label_encoder.inverse_transform(idxs)
                    suggestions = [w for w in all_suggestions if w.startswith(buffer_str)]
                    suggestions = sorted(suggestions, key=lambda w: (w != buffer_str, w))
                    if not suggestions:
                        all_words = list(autocomplete_label_encoder.classes_)
                        suggestions = [w for w in all_words if w.startswith(buffer_str)]
                        suggestions = sorted(suggestions, key=lambda w: (w != buffer_str, w))
                    if suggestions and idx < len(suggestions):
                        applied_words.append(suggestions[idx])
                        gesture_buffer.clear()
                except Exception as e:
                    print("LSTM suggestion error:", e)
        elif key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            # Nhận ký tự pending hiện tại (luôn cho phép thêm, kể cả trùng)
            if last_prediction:
                gesture_buffer.append(last_prediction)
                if len(gesture_buffer) > BUFFER_MAXLEN:
                    gesture_buffer.pop(0)
        elif key == 13:  # ENTER
            buffer_str = ''.join(gesture_buffer).upper()
            if len(buffer_str) >= 1:
                try:
                    topN = 200
                    seq = [char_dict.get(c, 0) for c in list(buffer_str)]
                    seq = pad_sequences([seq], maxlen=maxlen, padding='pre')
                    pred = predict_tflite(seq)
                    idxs = np.argsort(pred[0])[-topN:][::-1]
                    all_suggestions = autocomplete_label_encoder.inverse_transform(idxs)
                    suggestions = [w for w in all_suggestions if w.startswith(buffer_str)]
                    # Ưu tiên từ trùng buffer lên đầu
                    suggestions = sorted(suggestions, key=lambda w: (w != buffer_str, w))
                    if not suggestions:
                        all_words = list(autocomplete_label_encoder.classes_)
                        suggestions = [w for w in all_words if w.startswith(buffer_str)]
                        suggestions = sorted(suggestions, key=lambda w: (w != buffer_str, w))
                    if suggestions:
                        applied_words.append(suggestions[0])
                        gesture_buffer.clear()
                except Exception as e:
                    print("LSTM suggestion error:", e)
        elif key == 8:  # BACKSPACE
            if len(gesture_buffer) > 0:
                gesture_buffer.pop()

        # ==== HIỂN THỊ TEXT GỌN GÀNG, KHÔNG CHỒNG ====
        buffer_text = f"Buffer: {''.join(gesture_buffer)}"
        pending_text = f"Pending: {last_prediction if last_prediction else ''}"
        y_base = 50
        cv2.putText(frame, pending_text, (40, y_base), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
        if 'confidence' in locals():
            conf_text = f"Confidence: {confidence*100:.1f}%"
            cv2.putText(frame, conf_text, (40, y_base+40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,128,255), 2)
        cv2.putText(frame, buffer_text, (40, y_base+80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        # Hiển thị gợi ý (1-5) trên cùng một hàng, đánh số tương ứng phím bấm
        if 'suggestions' in locals() and suggestions:
            suggest_row = []
            for i, w in enumerate(suggestions[:5]):
                suggest_row.append(f"{i+1}. {w}")
            suggest_text = "   ".join(suggest_row)
            cv2.putText(frame, suggest_text, (40, y_base+120), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0,0,255), 2)

        # Output ở dưới cùng
        applied_text = f"Output: {' '.join(applied_words)}"
        h = frame.shape[0]
        cv2.putText(frame, applied_text, (40, h-30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)

        input_data = extract_landmarks(frame)

        if input_data is not None:
            # Reshape dữ liệu cho model
            input_data = input_data.reshape(1, SEQUENCE_LENGTH, NUM_LANDMARKS)

            # Dự đoán bằng TFLite
            gesture_interpreter.set_tensor(gesture_input_details[0]['index'], input_data.astype(np.float32))
            gesture_interpreter.invoke()
            prediction = gesture_interpreter.get_tensor(gesture_output_details[0]['index'])[0]
            predicted_label = np.argmax(prediction)
            confidence = prediction[predicted_label]

            # Kiểm tra xem predicted_label có trong label_map không
            if predicted_label in label_map:
                current_prediction = label_map[predicted_label]

                # Chỉ hiển thị dự đoán khi độ tin cậy cao
                if confidence > confidence_min:
                    if current_prediction == last_prediction:
                        prediction_counter += 1
                    else:
                        prediction_counter = 0
                        last_prediction = current_prediction

                    if prediction_counter >= prediction_threshold:
                        # Không tự động thêm gesture vào buffer nữa
                        display_text = f"Pending: {current_prediction if current_prediction else ''}"
                        confidence_text = f"Confidence: {confidence * 100:.1f}%"
                        buffer_text = f"Buffer: {''.join(gesture_buffer)}"
                        # Gợi ý top 5 từ bằng LSTM, hiển thị nổi bật top 1
                        buffer_str = ''.join(gesture_buffer).upper()
                        if len(buffer_str) >= 1:
                            try:
                                topN = 200  # Tăng số lượng từ được xét để lọc gợi ý nhiều hơn
                                seq = [char_dict.get(c, 0) for c in list(buffer_str)]
                                seq = pad_sequences([seq], maxlen=maxlen, padding='pre')
                                pred = predict_tflite(seq)
                                idxs = np.argsort(pred[0])[-topN:][::-1]
                                all_suggestions = autocomplete_label_encoder.inverse_transform(idxs)
                                # Lọc các từ bắt đầu bằng buffer
                                suggestions = [w for w in all_suggestions if w.startswith(buffer_str)]
                                # Ưu tiên từ trùng buffer lên đầu
                                suggestions = sorted(suggestions, key=lambda w: (w != buffer_str, w))
                                # Fallback: nếu không có từ nào, lấy từ toàn bộ label_encoder
                                if not suggestions:
                                    all_words = list(autocomplete_label_encoder.classes_)
                                    suggestions = [w for w in all_words if w.startswith(buffer_str)]
                                    suggestions = sorted(suggestions, key=lambda w: (w != buffer_str, w))
                                if suggestions:
                                    pass  # Đã có khối hiển thị suggest ở trên, không vẽ ở đây nữa
                            except Exception as e:
                                print("LSTM suggestion error:", e)
                        # Vẽ kết quả
                        pass  # Đã có khối hiển thị text duy nhất ở ngoài, không vẽ ở đây nữa
                else:
                    pass  # Không vẽ text ở đây
            else:
                pass  # Không vẽ text ở đây
        # Hiển thị frame
        cv2.imshow("Hand Gesture Recognition", frame)

    cap.release()
    cv2.destroyAllWindows()


def suggest_word_lstm(prefix):
    seq = [char_dict.get(c, 0) for c in list(prefix.upper())]
    seq = pad_sequences([seq], maxlen=maxlen, padding='pre')
    pred = predict_tflite(seq)
    idx = np.argmax(pred)
    return autocomplete_label_encoder.inverse_transform([idx])[0]


def predict_tflite(seq):
    # seq: numpy array shape (1, maxlen)
    # Đảm bảo đúng dtype float32
    autocomplete_interpreter.set_tensor(autocomplete_input_details[0]['index'], seq.astype(np.float32))
    autocomplete_interpreter.invoke()
    output = autocomplete_interpreter.get_tensor(autocomplete_output_details[0]['index'])
    return output


if __name__ == "__main__":
    main()