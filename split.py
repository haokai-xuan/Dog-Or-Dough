import shutil
import random
from pathlib import Path

# Path to the folder containing your images
image_folder = Path("./bread_loaves_images")

# Destination folders
output_folder = Path("./data")
cls = "dough"
train_folder = output_folder / f"train/{cls}"
val_folder = output_folder / f"val/{cls}"

# Create folders if they don't exist
for folder in [train_folder, val_folder]:
    folder.mkdir(parents=True, exist_ok=True)

# Get list of all images
images = [f for f in image_folder.rglob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]

# Shuffle for randomness
random.shuffle(images)

# Define split ratios
train_ratio = 0.7
val_ratio = 0.3

# Compute split indices
num_images = len(images)
train_end = int(train_ratio * num_images)

# Split images
train_images = images[:train_end]
val_images = images[train_end:]

# Move images to folders
for img in train_images:
    shutil.copy(img, train_folder)

for img in val_images:
    shutil.copy(img, val_folder)

print("Done! Train:", len(train_images), "Val:", len(val_images))
