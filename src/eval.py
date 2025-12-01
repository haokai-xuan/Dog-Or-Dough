import os
import numpy as np
from model import NeuralNetwork
from helpers.utils import preprocess_image, load_dataset
from helpers.metrics import get_stats, print_stats

# Configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "../checkpoints/Exp3/model_parameters_86.npz")

def inference(model, image_paths, class_names):
    X_list = []
    for img_path in image_paths:
        X_list.append(preprocess_image(img_path, target_size=(model.input_size[0], model.input_size[1])))
    X = np.hstack(X_list)
    predictions, probabilities = model.predict(X)

    results = []
    for i, img_path in enumerate(image_paths):
        pred_idx = predictions[i]
        pred_class = class_names[pred_idx]
        confidence = probabilities[pred_idx, i]

        results.append({
            "image_path": img_path,
            "predicted_class": pred_class,
            "predicted_idx": int(pred_idx),
            "confidence": float(confidence),
            "probabilities": {class_names[j]: float(probabilities[j, i]) for j in range(len(class_names))}
        })

    return results

def evaluate(model, test_dir, class_names):
    X_test, Y_test = load_dataset(test_dir, target_size=(model.input_size[0], model.input_size[1]))
    predictions, probabilities = model.predict(X_test)
    ground_truth = np.argmax(Y_test, axis=0)
    
    # DEBUG: Print what's happening
    print(f"DEBUG: Number of test samples: {len(ground_truth)}")
    print(f"DEBUG: Predictions shape: {predictions.shape}, unique values: {np.unique(predictions)}")
    print(f"DEBUG: Ground truth shape: {ground_truth.shape}, unique values: {np.unique(ground_truth)}")
    print(f"DEBUG: Predictions: {predictions[:10]}")  # First 10
    print(f"DEBUG: Ground truth: {ground_truth[:10]}")  # First 10
    print(f"DEBUG: Number of matches: {np.sum(predictions == ground_truth)}")
    print(f"DEBUG: Class names: {class_names}")
    
    stats = get_stats(predictions, ground_truth)
    
    return stats, predictions, ground_truth

def print_inference_results(results):
    print(f"\n{'='*60}")
    print(f"INFERENCE RESULTS ({len(results)} images)")
    print(f"{'='*60}\n")
    
    for result in results:
        print(f"Image: {os.path.basename(result['image_path'])}")
        print(f"Predicted: {result['predicted_class']} ({result['confidence']:.2%})")
        print(f"Probabilities:")
        for cls, prob in result['probabilities'].items():
            marker = " <--" if cls == result['predicted_class'] else ""
            print(f"  {cls}: {prob:.2%}{marker}")
        print()

if __name__ == "__main__":
    NN = NeuralNetwork(load_path=model_path)
    classes = sorted([d for d in os.listdir(os.path.join(script_dir, "../data/train"))])

    mode = "evaluate"

    if mode == "inference":
        folder = os.path.join(script_dir, "../data/test")
        image_paths = []
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(folder, fname))
        results = inference(NN, image_paths, classes)
        print_inference_results(results)
    elif mode == "evaluate":
        test_dir = os.path.join(script_dir, "../data/test")
        stats, predictions, ground_truth = evaluate(NN, test_dir, classes)

        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}\n")
        print_stats(stats, classes)
