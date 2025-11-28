import tensorflow as tf
import kagglehub
from pathlib import Path
import os
import numpy as np
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, classification_report

# ---------------------------
# Step 1 — Get dataset location
# ---------------------------
root = Path(kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset"))

# The Kaggle dataset is nested twice
dataset_folder = root / "New Plant Diseases Dataset(Augmented)" / "New Plant Diseases Dataset(Augmented)"

train_dir = dataset_folder / "train"
val_dir = dataset_folder / "valid"
test_dir  = dataset_folder / "test"   # Test folder exists in this dataset

print("Train:", train_dir)
print("Valid:", val_dir)
print("Test :", test_dir)

# -----------------------------
# 2. Basic config
# -----------------------------
IMG_HEIGHT = 128
IMG_WIDTH  = 128
BATCH_SIZE = 32
EPOCHS     = 20  # you can increase later if training looks good

# -----------------------------
# 2. Data loading (with augmentation)
# -----------------------------
# Load training dataset
train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

# Get number of classes and class names BEFORE applying map operations
num_classes = len(train_ds_raw.class_names)
class_names = train_ds_raw.class_names
print("Number of classes:", num_classes)
print("Class names:", class_names[:5], "...")  # Print first 5 as sample

# Normalize pixel values to [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1.0/255.0)
train_ds = train_ds_raw.map(lambda x, y: (normalization_layer(x), y))

# Create data augmentation layer
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.1),  # ~25 degrees
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomZoom(0.1),
    layers.RandomFlip(mode="horizontal_and_vertical"),
])

# Apply augmentation to training data
def augment(x, y):
    return data_augmentation(x, training=True), y

train_ds = train_ds.map(augment)

# Load validation dataset (no augmentation)
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False   # important so labels line up with predictions
)

# Normalize validation data
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# -----------------------------
# 3. Define a basic CNN (from scratch)
# -----------------------------
def build_cnn(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=38):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model

model = build_cnn(num_classes=num_classes)
model.summary()

# -----------------------------
# 4. Compile the model
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# 5. Callbacks (optional but useful)
# -----------------------------
checkpoint_path = "plant_cnn_best.keras"
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True
    )
]

# -----------------------------
# 6. Train the model
# -----------------------------
history_cnn = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=1
)

# -----------------------------
# 7. Plot training vs validation curves
# -----------------------------
epochs_cnn = range(1, len(history_cnn.history["loss"]) + 1)

plt.figure(figsize=(12, 5))

# Loss curves
plt.subplot(1, 2, 1)
plt.plot(epochs_cnn, history_cnn.history["loss"], label="Training loss")
plt.plot(epochs_cnn, history_cnn.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Loss (Training vs Validation)")
plt.legend()

# Accuracy curves
plt.subplot(1, 2, 2)
plt.plot(epochs_cnn, history_cnn.history["accuracy"], label="Training accuracy")
plt.plot(epochs_cnn, history_cnn.history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Accuracy (Training vs Validation)")
plt.legend()

plt.tight_layout()
plt.show()

# -----------------------------
# 8. Evaluate on validation set
# -----------------------------
val_loss_cnn, val_acc_cnn = model.evaluate(val_ds, verbose=0)
print(f"Validation loss:     {val_loss_cnn:.4f}")
print(f"Validation accuracy: {val_acc_cnn:.4f}")

# -----------------------------
# 9. Extra metrics: F1, confusion matrix, report
# -----------------------------
# Collect true labels and predictions
y_true = []
y_pred_probs_list = []

for images, labels in val_ds:
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred_probs_list.append(model.predict(images, verbose=0))

# Convert predictions to numpy array
y_pred_probs_cnn = np.vstack(y_pred_probs_list)
y_pred_labels_cnn = np.argmax(y_pred_probs_cnn, axis=1)
y_true = np.array(y_true)

# Macro F1-score
f1_cnn = f1_score(y_true, y_pred_labels_cnn, average="macro")
print(f"Validation macro F1-score: {f1_cnn:.4f}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_labels_cnn)
print("Confusion matrix:\n", cm)

# Classification report (per-class precision/recall/F1)
print("\nClassification report:")
print(classification_report(y_true, y_pred_labels_cnn, target_names=class_names))
