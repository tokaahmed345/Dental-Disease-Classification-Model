import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import matplotlib.pyplot as plt


root_dir = r'C:/Users/owner/Desktop/model/Dataset'
classes = ['Calculus', 'Gingivitis', 'Mouth_Ulcer', 'Caries']

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("🔹 Step 1: Checking dataset split...")

already_split = all(
    os.path.exists(os.path.join(root_dir, split, cls)) and
    len(os.listdir(os.path.join(root_dir, split, cls))) > 0
    for split in ['train','val','test'] for cls in classes
)

if already_split:
    print("✅ Dataset already split. Skipping this step.")
else:
    print("⚡ Splitting dataset into train/val/test...")
    for split in ['train','val','test']:
        for cls in classes:
            os.makedirs(os.path.join(root_dir, split, cls), exist_ok=True)

    train_ratio = 0.7
    val_ratio = 0.15

    for cls in classes:
        src_path = os.path.join(root_dir, cls)
        all_images = os.listdir(src_path)
        np.random.shuffle(all_images)

        train_count = int(len(all_images) * train_ratio)
        val_count = int(len(all_images) * val_ratio)

        train_files = all_images[:train_count]
        val_files = all_images[train_count:train_count+val_count]
        test_files = all_images[train_count+val_count:]

        for file in train_files:
            shutil.copy(os.path.join(src_path, file), os.path.join(root_dir, 'train', cls, file))
        for file in val_files:
            shutil.copy(os.path.join(src_path, file), os.path.join(root_dir, 'val', cls, file))
        for file in test_files:
            shutil.copy(os.path.join(src_path, file), os.path.join(root_dir, 'test', cls, file))

    print("✅ Dataset splitted into train/val/test successfully!")

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    shear_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    os.path.join(root_dir, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(root_dir, 'val'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(classes), activation='softmax')
])


model.compile(
    optimizer=optimizers.Adam(3e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n🚀 Phase 1: Training classifier head...")
model.fit(
    train_generator,
    epochs=12,
    validation_data=val_generator
)


print("\n🔓 Fine-Tuning MobileNetV2...")

base_model.trainable = True
fine_tune_at = int(len(base_model.layers) * 0.7)
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=8,
    restore_best_weights=True
)

history = model.fit(
    train_generator,
    epochs=40,
    validation_data=val_generator,
    callbacks=[early_stopping]
)


model.save('dental_v2.h5')
print("\n✅ Model saved as dental_v2.h5")

plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Dental Disease Classification Performance')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()