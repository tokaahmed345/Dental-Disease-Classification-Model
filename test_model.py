import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt


MODEL_PATH = 'dental_v2.h5'
CLASS_NAMES = ['Calculus', 'Caries', 'Gingivitis', 'Mouth_Ulcer']

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
else:
    print(f"❌ Error: {MODEL_PATH} not found!")


def predict_any_image(img_path):
    if not os.path.exists(img_path):
        print(f"❌ File not found at: {img_path}")
        return

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)

    predictions = model.predict(img_preprocessed)
    
    best_index = np.argmax(predictions[0])
    label = CLASS_NAMES[best_index]
    confidence = predictions[0][best_index] * 100

    print("-" * 30)
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2f}%")
    print("-" * 30)

    plt.imshow(img)
    plt.title(f"{label} ({confidence:.1f}%)")
    plt.axis('off')
    plt.show()


test_image = r'C:/Users/owner/Desktop/test_image.jfif'

predict_any_image(test_image)
# 'C:/Users/owner/Desktop/test_image.jfif'