import numpy as np
import os
import csv
import tensorflow as tf
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Giữ nguyên sequence length
SEQUENCE_LENGTH = 30
NUM_LANDMARKS = 21 * 2


def load_or_create_label_mapping():
    """Load label mapping từ file hoặc tạo mới nếu chưa có"""
    try:
        with open('label_mapping.json', 'r') as f:
            label_map = json.load(f)
            # Chuyển đổi string keys thành integers
            label_map = {int(k): v for k, v in label_map.items()}
            return label_map
    except FileNotFoundError:
        # Tạo mapping mặc định cho 26 chữ cái
        label_map = {i: chr(65 + i) for i in range(26)}
        # Lưu mapping mới
        with open('label_mapping.json', 'w') as f:
            json.dump(label_map, f)
        return label_map


def save_label_mapping(label_map):
    """Lưu label mapping vào file"""
    with open('label_mapping.json', 'w') as f:
        json.dump(label_map, f)


# Load dữ liệu
DATA_DIR = '.'
labels = []
data = []
label_map = load_or_create_label_mapping()

print("Loading dataset...")
for file in os.listdir(DATA_DIR):
    if file.startswith("hand_gesture_") and file.endswith(".csv"):
        label = file.split("_")[-1].split(".")[0].upper()

        # Thêm label mới vào mapping nếu chưa có
        if label not in label_map.values():
            new_index = max(label_map.keys()) + 1
            label_map[new_index] = label
            print(f"Added new label: {label} with index {new_index}")

        # Tìm index của label trong mapping
        label_index = [k for k, v in label_map.items() if v == label][0]

        with open(os.path.join(DATA_DIR, file), 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == SEQUENCE_LENGTH * NUM_LANDMARKS:
                    data.append(np.array(row, dtype=np.float32))
                    labels.append(label_index)

print(f"Total samples loaded: {len(data)}")
print(f"Current label mapping: {label_map}")

if len(data) == 0:
    raise ValueError("No valid data found!")

# Chuyển đổi dữ liệu
X = np.array(data).reshape(-1, SEQUENCE_LENGTH, NUM_LANDMARKS)
y = to_categorical(labels, num_classes=len(label_map))


# Data augmentation đơn giản hơn
def augment_data(X, y):
    augmented_X = []
    augmented_y = []

    for i in range(len(X)):
        # Mẫu gốc
        augmented_X.append(X[i])
        augmented_y.append(y[i])

        # Chỉ thêm một augmentation đơn giản
        noise = np.random.normal(0, 0.005, X[i].shape)
        augmented_X.append(X[i] + noise)
        augmented_y.append(y[i])

    return np.array(augmented_X), np.array(augmented_y)


# Chia tập train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Augment training data
X_train_aug, y_train_aug = augment_data(X_train, y_train)


# Model đơn giản hơn
def create_model():
    model = Sequential([
        # First LSTM layer
        LSTM(256, return_sequences=True, activation='relu',
             input_shape=(SEQUENCE_LENGTH, NUM_LANDMARKS)),
        BatchNormalization(),
        Dropout(0.3),

        # Second LSTM layer
        LSTM(256, return_sequences=True, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        # Third LSTM layer
        LSTM(256, return_sequences=True, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        # Fourth LSTM layer
        LSTM(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        # Dense layers
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        BatchNormalization(),

        Dense(len(label_map), activation='softmax')
    ])
    return model


# Tạo và compile model
model = create_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Thêm các callback
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        mode='max'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.2,
        patience=7,
        min_lr=0.00001,
        mode='max'
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
]

# Training với batch size nhỏ hơn
history = model.fit(
    X_train_aug, y_train_aug,
    validation_data=(X_test, y_test),
    epochs=75,
    batch_size=32,
    callbacks=callbacks,
    shuffle=True
)

# Lưu model và label mapping
model.save("lstm_hand_gesture_model.keras")
save_label_mapping(label_map)
print("Model and label mapping saved successfully!")
print(f"Final label mapping: {label_map}")

# Đánh giá model (Load model tốt nhất đã lưu nếu cần)
# Thay vì dùng model cuối cùng, hãy load model tốt nhất đã được lưu bởi ModelCheckpoint
print("\nEvaluating the best model saved...")
best_model = tf.keras.models.load_model('best_model.keras') 
test_loss, test_acc = best_model.evaluate(X_test, y_test)
print(f"Best model test accuracy: {test_acc:.4f}")

# In confusion matrix và classification report với best_model
print("\nGenerating reports for the best model...")
y_pred = best_model.predict(X_test) # Sử dụng best_model
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Confusion Matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Best Model)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix_best_model.png') # Lưu file ảnh mới
plt.close()

# Classification Report
print("\nDetailed Classification Report (Best Model):")
target_names = [label_map[i] for i in range(len(label_map))]
print(classification_report(y_test_classes, y_pred_classes, target_names=target_names))
