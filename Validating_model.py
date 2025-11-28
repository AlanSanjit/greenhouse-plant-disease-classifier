import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from keras.utils import load_img, img_to_array
import kagglehub


# -----------------------------------------
# User settings
# -----------------------------------------
root = Path(kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset"))

# The Kaggle dataset is nested twice
dataset_folder = root / "New Plant Diseases Dataset(Augmented)" / "New Plant Diseases Dataset(Augmented)"

# Use the labeled validation split for evaluation
eval_dir = dataset_folder / "valid"
MODEL_PATH = Path("plant_cnn_best.keras")

print("Evaluation directory:", eval_dir)
IMG_SIZE = 128
BATCH_SIZE = 32

# -----------------------------------------
# 1. Load the trained model
# -----------------------------------------
print("Loading model:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!\n")

# -----------------------------------------
# 2. Load evaluation dataset
# -----------------------------------------
print("Loading evaluation dataset...")
test_ds_raw = tf.keras.utils.image_dataset_from_directory(
    eval_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

# Extract class names before mapping (since .map removes attribute)
class_names = test_ds_raw.class_names
num_classes = len(class_names)
print("\nClasses:", class_names)
print("Number of classes:", num_classes)

# Normalize dataset
normalization = tf.keras.layers.Rescaling(1/255.0)
test_ds = test_ds_raw.map(lambda x, y: (normalization(x), y))

# -----------------------------------------
# 3. Evaluate model on evaluation set
# -----------------------------------------
print("\nEvaluating model...")
test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print(f"\nEval Loss: {test_loss:.4f}")
print(f"Eval Accuracy: {test_acc:.4f}\n")

# -----------------------------------------
# 4. Classification report + confusion matrix
# -----------------------------------------
y_true = []
y_pred_probs = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_pred_probs.append(preds)
    y_true.extend(np.argmax(labels.numpy(), axis=1))

y_pred_probs = np.vstack(y_pred_probs)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.array(y_true)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# -----------------------------------------
# 5. Predict on a single image (demo)
# -----------------------------------------

def predict_single_image(image_path):
    print("\nPredicting on single image:", image_path)
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    x = img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    pred_idx = np.argmax(preds)
    confidence = preds[0][pred_idx]

    print(f"Prediction: {class_names[pred_idx]} (confidence: {confidence:.3f})")

# Example use (uncomment and put a real file path)
# predict_single_image("leaf.jpg")