import numpy as np
import matplotlib.pyplot as plt
import os

def get_stats(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    classes = np.unique(y_true)

    per_class_precision = {}
    per_class_recall = {}
    per_class_f1 = {}

    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class_precision[c] = precision
        per_class_recall[c] = recall
        per_class_f1[c] = f1

    macro_precision = np.mean(list(per_class_precision.values()))
    macro_recall = np.mean(list(per_class_recall.values()))
    macro_f1 = np.mean(list(per_class_f1.values()))

    accuracy = np.mean(y_pred == y_true)

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1
    }

def plot_loss(train_losses, val_losses, title="Train and Val Loss Over Epochs", save_path="loss_curve.png"):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs, val_losses, label="Val Loss", marker="o")

    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def print_stats(stats, classes):
    print("STATS")

    print(f"Accuracy: {stats["accuracy"]}")
    print("")

    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 50)

    for i, cls in enumerate(classes):
        if i in stats["per_class_precision"]:
            precision = stats["per_class_precision"][i]
            recall = stats["per_class_recall"][i]
            f1 = stats["per_class_f1"][i]
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
            
        print(f"{str(cls):<15} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")

    print("")
    
    print(f"Macro precision: {stats["macro_precision"]:.4f}")
    print(f"Macro recall: {stats["macro_recall"]:.4f}")
    print(f"Macro F1: {stats["macro_f1"]:.4f}")