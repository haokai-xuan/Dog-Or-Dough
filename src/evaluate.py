import os
import numpy as np
from model import NeuralNetwork
from helpers.utils import preprocess_image

# Configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "../checkpoints/Exp1/model_parameters_62.npz")
folder_paths = os.path.join(script_dir, "../data/test")

# Get all image paths
if isinstance(folder_paths, str):
    folder_paths = [folder_paths]

image_paths = []
for folder in folder_paths:
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))

# Preprocess images
X_list = []
valid_paths = []
for img_path in image_paths:
    try:
        X_list.append(preprocess_image(img_path, target_size=(64, 64)))
        valid_paths.append(img_path)
    except:
        continue

X = np.hstack(X_list)

# Load model and predict
NN = NeuralNetwork(load_path=model_path)

predictions, probabilities = NN.predict(X)

# Get class names
classes = sorted([d for d in os.listdir(os.path.join(script_dir, "../data/train")) 
                 if os.path.isdir(os.path.join(script_dir, "../data/train", d))])

# Print results
print(f"\nEvaluated {len(valid_paths)} images:\n")
for i, img_path in enumerate(valid_paths):
    pred_idx = predictions[i]
    pred_class = classes[pred_idx]
    confidence = probabilities[pred_idx, i]
    print(f"{os.path.basename(img_path)}: {pred_class} ({confidence:.2%})")
