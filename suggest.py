import requests
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import tensorflow as tf
import os
from collections import Counter

# Đọc bộ từ Oxford 3000
try:
    with open('oxford3000.txt', 'r') as f:
        oxford_words = [w.strip().upper() for w in f.readlines() if w.strip()]
except FileNotFoundError:
    raise FileNotFoundError("Không tìm thấy file oxford3000.txt. Vui lòng tải bộ từ Oxford 3000 về cùng thư mục!")

# Lấy bộ từ phổ biến từ Google 20k English Words
url = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/20k.txt"
response = requests.get(url)
google_words = [w.strip().upper() for w in response.text.splitlines() if w.isalpha() and len(w) <= 12]

daily_words = []
if os.path.exists('daily_conversation.txt'):
    with open('daily_conversation.txt', 'r') as f:
        daily_words = [w.strip().upper() for w in f.readlines() if w.strip()]

# Kết hợp, ưu tiên oxford, rồi daily, rồi google, không trùng lặp
all_words = oxford_words + [w for w in daily_words if w not in oxford_words] + [w for w in google_words if w not in oxford_words and w not in daily_words]

# Tính tần suất xuất hiện của từng từ trong 3 file
word_freq = Counter()
for w in oxford_words:
    word_freq[w] += 1
for w in daily_words:
    word_freq[w] += 1
for w in google_words:
    word_freq[w] += 1

# Lấy top 1000 từ có tần suất cao nhất, ưu tiên oxford, rồi daily, rồi google khi bằng tần suất
def word_priority_key(w):
    # Ưu tiên oxford trước, rồi daily, rồi google
    if w in oxford_words:
        return (0, -word_freq[w], oxford_words.index(w))
    elif w in daily_words:
        return (1, -word_freq[w], daily_words.index(w))
    else:
        return (2, -word_freq[w], google_words.index(w) if w in google_words else 99999)

all_unique = list(word_freq.keys())
top1000 = sorted(all_unique, key=word_priority_key)[:1000]

# Lặp lại mỗi từ đúng bằng số tần suất (1, 2 hoặc 3 lần)
priority_list = []
for w in top1000:
    priority_list.extend([w] * word_freq[w])

# Sau đó thêm các từ còn lại (không ưu tiên) vào
other_words = [w for w in all_words if w not in top1000]
all_words_priority = priority_list + other_words
unique_words = all_words_priority[:10000]

# ===== 2. Tạo tập dữ liệu prefix → word =====
input_seqs = []
target_words = []
for word in unique_words:
    for i in range(1, len(word) + 1):
        prefix = word[:i]
        input_seqs.append(list(prefix))
        target_words.append(word)

# ===== 3. Mã hóa ký tự =====
char_dict = {ch: i+1 for i, ch in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}
maxlen = 12  # độ dài chuỗi đầu vào (tăng lên cho phù hợp từ dài)

X_encoded = [[char_dict.get(c, 0) for c in seq] for seq in input_seqs]
X_padded = pad_sequences(X_encoded, maxlen=maxlen, padding='pre')

# ===== 4. Mã hóa nhãn bằng số nguyên =====
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(target_words)
vocab_size = len(label_encoder.classes_)

# ===== 5. Xây dựng mô hình LSTM =====
model = Sequential()
model.add(Embedding(input_dim=27, output_dim=128, input_length=maxlen))  # 26 chữ + 1 padding
model.add(LSTM(256))
model.add(Dense(vocab_size, activation='softmax'))

# ===== 6. Compile và train =====
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_padded, y_encoded, epochs=30, batch_size=256, validation_split=0.1)

# ===== 7. Lưu model và label_encoder =====
model.save('lstm_word_autocomplete.h5')
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

if __name__ == "__main__":
    print("Đã train và lưu model, label_encoder thành công!")