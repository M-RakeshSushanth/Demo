import tensorflow as tf
from keras.models import Sequential
from keras.layers import (
    Conv2D, MaxPooling2D, Dense,
    Flatten, Dropout, BatchNormalization
)
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------
# PARAMETERS
# -----------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30   # slightly increased, not too much

# -----------------------
# DATA GENERATORS
# -----------------------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# -----------------------
# LOAD TRAIN DATA
# -----------------------
train_data = train_datagen.flow_from_directory(
    "dataset/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# -----------------------
# LOAD TEST DATA (IMPORTANT FIX)
# -----------------------
test_data = test_datagen.flow_from_directory(
    "dataset/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False   # VERY IMPORTANT
)

# -----------------------
# CNN MODEL (ADJUSTED)
# -----------------------
model = Sequential([
    Conv2D(32, (3, 3), activation="relu",
           input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation="relu"),   # increased capacity
    Dropout(0.5),

    Dense(3, activation="softmax")
])

# -----------------------
# COMPILE MODEL
# -----------------------
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------
# TRAIN MODEL
# -----------------------
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=test_data
)

# -----------------------
# SAVE MODEL
# -----------------------
model.save("osteoporosis_3class_model.h5")

# -----------------------
# SAVE ACCURACY GRAPH
# -----------------------
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.savefig("training_validation_accuracy.png", dpi=300)
plt.show()

# -----------------------
# SAVE ACCURACY VALUES
# -----------------------
df = pd.DataFrame({
    "epoch": range(1, EPOCHS + 1),
    "train_accuracy": history.history["accuracy"],
    "val_accuracy": history.history["val_accuracy"]
})
df.to_csv("training_accuracy_values.csv", index=False)
