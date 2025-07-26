import tensorflow as tf

# Load mô hình Keras đã train
model = tf.keras.models.load_model('lstm_word_autocomplete.h5')

# Chuyển sang TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter._experimental_lower_tensor_list_ops = False
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # TFLite ops
    tf.lite.OpsSet.SELECT_TF_OPS     # Cho phép một số ops của TF gốc
]
converter._experimental_lower_tensor_list_ops = False
tflite_model = converter.convert()

# Lưu file TFLite
with open('lstm_word_autocomplete.tflite', 'wb') as f:
    f.write(tflite_model)

print("Đã chuyển đổi sang lstm_word_autocomplete.tflite thành công!")

# Chuyển model lstm_hand_gesture_model.keras sang TFLite
print("\nConverting lstm_hand_gesture_model.keras to TFLite...")
gesture_model = tf.keras.models.load_model('lstm_hand_gesture_model.keras')
gesture_converter = tf.lite.TFLiteConverter.from_keras_model(gesture_model)
gesture_converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
gesture_converter._experimental_lower_tensor_list_ops = False
gesture_converter.experimental_enable_resource_variables = True
gesture_tflite_model = gesture_converter.convert()
with open('lstm_hand_gesture_model.tflite', 'wb') as f:
    f.write(gesture_tflite_model)
print("TFLite model saved as lstm_hand_gesture_model.tflite")