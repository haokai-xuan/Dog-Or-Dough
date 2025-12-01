import numpy as np
from tqdm import tqdm
import os
from model import NeuralNetwork
from helpers.utils import load_dataset, get_image_paths, get_batches, get_augmentation_pipeline
from helpers.metrics import *


class Trainer:
    def __init__(self, NN, epochs=200, batch_size=32, patience=20, use_lr_decay=False, lr_decay=0.5, lr_patience=5, min_lr=1e-7, aug_pipeline=None, data_folder_name="data", ckpt_folder_name="checkpoints", experiment_name="Exp1"):
        self.NN = NN
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience

        self.use_lr_decay = use_lr_decay
        self.lr_decay = lr_decay
        self.lr_patience = lr_patience
        self.min_lr = min_lr
        self.aug_pipeline = aug_pipeline

        self.script_directory = os.path.dirname(__file__)
        self.ckpt_folder_name = ckpt_folder_name
        self.experiment_name = experiment_name
        os.makedirs(f"{self.ckpt_folder_name}/{self.experiment_name}", exist_ok=True)

        self.X_paths, self.Y = get_image_paths(os.path.join(self.script_directory, f"../{data_folder_name}/train"))
        self.val_X, self.val_Y = load_dataset(os.path.join(self.script_directory, f"../{data_folder_name}/val"))

        self.classes = sorted(os.listdir(os.path.join(self.script_directory, "../data/train")))

        self._print_parameters()

    def _print_parameters(self):
        layer_dims = [self.NN.input_size[0] * self.NN.input_size[1] * (3 if self.NN.color else 1)] + self.NN.layers
        num = 0
        for i in range(1, len(layer_dims)):
            num += layer_dims[i] * layer_dims[i - 1] + layer_dims[i]
        print(f"Trainable parameters: {num}")

    def _evaluate(self, X, Y):
        predictions, AL = self.NN.predict(X)
        ground_truths = np.argmax(Y, axis=0)
        cost = self.NN.compute_cost(AL, Y)
        return predictions, ground_truths, cost

    def train(self):
        best_f1 = float("-inf")
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        epochs_plateau = 0
        t = 1
        train_losses, val_losses = [], []

        for epoch in tqdm(range(self.epochs), desc="Epochs"):
            batches = get_batches(self.X_paths, self.Y, self.batch_size, aug_pipeline=self.aug_pipeline, target_size=self.NN.input_size)
            epoch_train_losses = []

            for X, Y in tqdm(batches, desc="Batches", leave=False):
                AL, caches = self.NN.forward(X)
                train_loss = self.NN.compute_cost(AL, Y)
                epoch_train_losses.append(train_loss)
                grads = self.NN.backward(AL, Y, caches)
                self.NN.update_parameters(grads, t)

                t += 1

            predictions, ground_truth, val_loss = self._evaluate(self.val_X, self.val_Y)
            stats = get_stats(predictions, ground_truth)

            avg_train_loss = np.mean(epoch_train_losses)
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)

            if stats["macro_f1"] > best_f1:
                best_f1 = stats["macro_f1"]
                self.NN.save_model(os.path.join(self.script_directory, f"../{self.ckpt_folder_name}/{self.experiment_name}/model_parameters_{epoch}.npz"))

            print("=" * 60)
            print(f"Epoch {epoch + 1}/{self.epochs}")
            print("=" * 60)
            print(f"Train loss: {avg_train_loss:.4f}")
            print(f"Val loss: {val_loss:.4f}\n")
            print_stats(stats, self.classes)
            plot_loss(train_losses, val_losses, save_path=os.path.join(self.script_directory, f"../{self.ckpt_folder_name}/{self.experiment_name}/loss_curve.png"))

            # Track improvement
            if val_loss < best_val_loss:  
                best_val_loss = val_loss
                epochs_without_improvement = 0
                epochs_plateau = 0
            else:
                epochs_without_improvement += 1
                epochs_plateau += 1

                # ---- Learning Rate Decay ----
                if self.use_lr_decay and epochs_plateau >= self.lr_patience:
                    old_lr = self.NN.learning_rate
                    self.NN.learning_rate = max(self.min_lr, old_lr * self.lr_decay)
                    print(f"⚠️ Learning rate decayed: {old_lr} → {self.NN.learning_rate}")
                    epochs_plateau = 0

                # ---- Early Stopping ----
                if epochs_without_improvement >= self.patience:
                    print(f"🛑 Early stopping triggered after {epochs_without_improvement} epochs without improvement.")
                    break

            print(f"Epochs without improvement: {epochs_without_improvement}")

            print("")
            print("")
            print("")

        print("Training Finished.")


NN = NeuralNetwork(
    layers=[64, 32, 32, 16, 8, 2],
    input_size=(64, 64),
    learning_rate=1e-4,
    weight_decay=1e-5,
    dropout=[0.3, 0.2, 0.2, 0.1, 0.1]
)
trainer = Trainer(
    NN=NN,
    use_lr_decay=True,
    lr_patience=5,
    experiment_name="Exp3",
)
trainer.train()