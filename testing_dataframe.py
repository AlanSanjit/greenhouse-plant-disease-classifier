import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
import kagglehub

# ---------------------------
# User settings
# ---------------------------
MODEL_PATH = "plant_cnn_best.keras"   # change if you used a different file name
IMG_SIZE = 128
BATCH_SIZE = 32

# ---------------------------
# 1. Get dataset location
# ---------------------------
root = Path(kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset"))

# The Kaggle dataset is nested twice
dataset_folder = root / "New Plant Diseases Dataset(Augmented)" / "New Plant Diseases Dataset(Augmented)"
train_dir = dataset_folder / "train"
test_dir = dataset_folder / "test"

print("Test directory:", test_dir)

# ---------------------------
# 2. Load the trained model
# ---------------------------
print("\nLoading model:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# ---------------------------
# 3. Build test generator
# ---------------------------
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

test_datagen = ImageDataGenerator(rescale=1./255.0)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode=None,            # no labels in Kaggle test set
    shuffle=False               # IMPORTANT: so filenames align with predictions
)

# Class name ↔ index mapping (derived from train directory to match training order)
class_names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
class_to_idx = {name: idx for idx, name in enumerate(class_names)}
idx_to_class = {idx: name for name, idx in class_to_idx.items()}

print("\nNumber of classes (from train set):", len(class_names))
print("Example classes:", class_names[:5])

# ---------------------------
# 4. Run predictions on test set
# ---------------------------
print("\nRunning predictions on test set...")
pred_probs = model.predict(test_generator, verbose=1)  # shape: (num_samples, num_classes)
pred_class_indices = np.argmax(pred_probs, axis=1)
pred_confidences = np.max(pred_probs, axis=1)

# Filenames (relative to test_dir)
filenames = test_generator.filenames                   # e.g. "Apple___Apple_scab/image_0001.jpg"

# ---------------------------
# 5. Build DataFrame for CSV
# ---------------------------
rows = []
for i, rel_path in enumerate(filenames):
    pred_idx = pred_class_indices[i]
    conf = float(pred_confidences[i])

    # infer "true" label from folder name when available (Kaggle test set has none)
    parts = Path(rel_path).parts
    inferred_true_name = parts[0] if len(parts) > 1 else None
    true_idx = class_to_idx.get(inferred_true_name, -1)

    pred_name = idx_to_class.get(pred_idx, f"class_{pred_idx}")

    rows.append({
        "relative_path": rel_path,
        "absolute_path": str(test_dir / rel_path),
        "true_class_index": int(true_idx),
        "true_class_name": inferred_true_name,
        "pred_class_index": int(pred_idx),
        "pred_class_name": pred_name,
        "pred_confidence": conf
    })

df = pd.DataFrame(rows)

# ---------------------------
# 6. Save to CSV
# ---------------------------
csv_path = "test_predictions.csv"
df.to_csv(csv_path, index=False)

print(f"\nSaved predictions for {len(df)} images to: {csv_path}")
print(df.head())