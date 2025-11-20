import tensorflow as tf
import kagglehub
from pathlib import Path

# ---------------------------
# Step 1 — Get dataset location
# ---------------------------
root = Path(kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset"))

# The Kaggle dataset is nested twice
dataset_folder = root / "New Plant Diseases Dataset(Augmented)" / "New Plant Diseases Dataset(Augmented)"

train_dir = dataset_folder / "train"
valid_dir = dataset_folder / "valid"
test_dir  = dataset_folder / "test"   # Test folder exists in this dataset

print("Train:", train_dir)
print("Valid:", valid_dir)
print("Test :", test_dir)

# ---------------------------
# Step 2 — Settings
# ---------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ---------------------------
# Step 3 — Load datasets
# ---------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    valid_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)
