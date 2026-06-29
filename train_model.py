import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = "dataset/Training"
test_dir = "dataset/Testing"

# Set to 128x128 as requested
img_size = (128, 128)
batch_size = 16

# 1. Initialize data loaders with real-time normalization
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

print("[*] Streaming training batches from directory...")
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

print("[*] Streaming validation batches from directory...")
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

# 2. Custom CNN Architecture Layout
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),

    tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(4, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 3. Model Optimization Execution Loop
print("[*] Starting custom CNN network training loop...")
model.fit(
    train_data,
    validation_data=test_data,
    epochs=5
)

# 4. Safe Directory Verification & Save Process
output_dir = "model"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, "brain_tumor_model.h5")

model.save(save_path)
print(f"Model saved successfully at: {save_path}")
