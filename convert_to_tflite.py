import tensorflow as tf
import os

model_path = 'dental_v2.h5'
model = tf.keras.models.load_model(model_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

tflite_file = 'dental_model.tflite'
with open(tflite_file, 'wb') as f:
    f.write(tflite_model)

print(f"✅ Model converted successfully: {tflite_file}")
classes = ['Calculus', 'Caries', 'Gingivitis', 'Mouth_Ulcer']
with open('labels.txt', 'w') as f:
    for item in classes:
        f.write(item + '\n')

print("✅ labels.txt file created successfully")