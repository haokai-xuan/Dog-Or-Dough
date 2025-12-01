import numpy as np
from helpers.activations import *

class NeuralNetwork:
    def __init__(
        self,
        layers=[64, 64, 32, 2],
        input_size=(64, 64),
        learning_rate=0.01,
        weight_decay=0.0,
        dropout=None,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        color=True,
        load_path=""
    ):
        if load_path:
            self._load_model(load_path)
        else:
            self.layers = layers
            self.input_size = input_size
            self.learning_rate = learning_rate
            self.weight_decay = weight_decay
            self.dropout = dropout
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            self.color = color

            self.parameters = {}
            self.cache = {}
            self.v = {} # First gradients moving average
            self.s = {} # Squared gradients moving average

            self._initialize_parameters()

    def _initialize_parameters(self):
        layer_dims = [self.input_size[0] * self.input_size[1] * (3 if self.color else 1)] + self.layers

        for l in range(1, len(layer_dims)):
            input_dim = layer_dims[l - 1]
            output_dim = layer_dims[l]

            W = np.random.randn(output_dim, input_dim) * np.sqrt(2 / layer_dims[l - 1])
            b = np.zeros((output_dim, 1))

            self.parameters["W" + str(l)] = W
            self.parameters["b" + str(l)] = b

            self.v["dW" + str(l)] = np.zeros_like(W)
            self.v["db" + str(l)] = np.zeros_like(b)
            self.s["dW" + str(l)] = np.zeros_like(W)
            self.s["db" + str(l)] = np.zeros_like(b)

    def _linear_forward(self, A_prev, W, b):
        Z = np.dot(W, A_prev) + b
        cache = (A_prev, W, b)
        return Z, cache

    def _linear_activation_forward(self, A_prev, W, b, activation):
        Z, linear_cache = self._linear_forward(A_prev, W, b)
        if activation == "relu":
            A, activation_cache = relu(Z)

        elif activation == "softmax":
            A, activation_cache = softmax(Z)

        cache = (linear_cache, activation_cache)

        return A, cache

    def forward(self, X, training=True):
        caches = []
        A = X
        L = len(self.layers)

        for l in range(1, L):
            A_prev = A
            A, cache = self._linear_activation_forward(
                A_prev, self.parameters["W" + str(l)], self.parameters["b" + str(l)], "relu"
            )
            D = None
            if self.dropout and training:
                D = np.random.rand(self.parameters["W" + str(l)].shape[0], X.shape[1])
                D = (D > self.dropout[l - 1]).astype(int)
                A = A * D
                A = A / (1 - self.dropout[l - 1])
            caches.append((*cache, D))

        AL, cache = self._linear_activation_forward(
            A, self.parameters["W" + str(L)], self.parameters["b" + str(L)], "softmax"
        )
        caches.append((*cache, None))
        return AL, caches

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cross_entropy_cost = -np.sum(Y * np.log(AL + self.epsilon)) / m
        frobenius_norm = 0
        for l in range(1, len(self.layers) + 1):
            W = self.parameters["W" + str(l)]
            frobenius_norm += np.sum(np.square(W))
        L2_cost = (self.weight_decay / (2 * m)) * frobenius_norm
        cost = cross_entropy_cost + L2_cost
        return np.squeeze(cost)

    def _linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = np.dot(dZ, A_prev.T) / m + (self.weight_decay / m) * W
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def _linear_activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache, D = cache
        if activation == "relu":
            dZ = relu_backward(dA, activation_cache)
        elif activation == "softmax":
            dZ = softmax_backward(dA, activation_cache)

        dA_prev, dW, db = self._linear_backward(dZ, linear_cache)
        return dA_prev, dW, db

    def backward(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        dAL = AL - Y
        current_cache = caches[L - 1]
        dA_prev_temp, dW_temp, db_temp = self._linear_activation_backward(
            dAL, current_cache, "softmax"
        )
        grads["dA" + str(L - 1)] = dA_prev_temp
        grads["dW" + str(L)] = dW_temp
        grads["db" + str(L)] = db_temp

        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            linear_cache, activation_cache, D = current_cache

            dA = grads["dA" + str(l + 1)]
            if D is not None:
                dA = dA * D
                dA = dA / (1 - self.dropout[l])

            dA_prev_temp, dW_temp, db_temp = self._linear_activation_backward(
                dA, (linear_cache, activation_cache, D), "relu"
            )
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def update_parameters(self, grads, t):
        L = len(self.parameters) // 2
        v_corrected = {}
        s_corrected = {}

        for l in range(1, L + 1):
            self.v["dW" + str(l)] = self.beta1 * self.v["dW" + str(l)] + (1 - self.beta1) * grads["dW" + str(l)]
            self.v["db" + str(l)] = self.beta1 * self.v["db" + str(l)] + (1 - self.beta1) * grads["db" + str(l)]

            v_corrected["dW" + str(l)] = self.v["dW" + str(l)] / (1 - self.beta1 ** t)
            v_corrected["db" + str(l)] = self.v["db" + str(l)] / (1 - self.beta1 ** t)

            self.s["dW" + str(l)] = self.beta2 * self.s["dW" + str(l)] + (1 - self.beta2) * grads["dW" + str(l)] ** 2
            self.s["db" + str(l)] = self.beta2 * self.s["db" + str(l)] + (1 - self.beta2) * grads["db" + str(l)] ** 2

            s_corrected["dW" + str(l)] = self.s["dW" + str(l)] / (1 - self.beta2 ** t)
            s_corrected["db" + str(l)] = self.s["db" + str(l)] / (1 - self.beta2 ** t)

            self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - self.learning_rate * v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + self.epsilon)
            self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - self.learning_rate * v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + self.epsilon)

    def predict(self, X):
        AL, _ = self.forward(X, training=False)
        predictions = np.argmax(AL, axis=0)
        return predictions, AL
    
    def save_model(self, path="model_parameters.npz"):
        np.savez(
            path,
            **self.parameters,
            layers=self.layers,
            input_size=self.input_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            dropout=self.dropout,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            color=self.color
        )
        print(f"Model saved to {path}")

    def _load_model(self, path="model_parameters.npz"):
        data = np.load(path, allow_pickle=True)
        self.parameters = {}

        for key in data.files:
            if key.startswith("W") or (key.startswith("b") and key[1].isdigit()):
                self.parameters[key] = data[key]

        if "layers" in data:
            self.layers = data["layers"].tolist() if isinstance(data["layers"], np.ndarray) else data["layers"]
            self.input_size = tuple(map(int, data["input_size"]))
            self.learning_rate = float(data["learning_rate"])
            self.weight_decay = float(data["weight_decay"])
            self.dropout = data["dropout"].tolist() if data["dropout"] is not None else None
            self.beta1 = float(data["beta1"])
            self.beta2 = float(data["beta2"])
            self.epsilon = float(data["epsilon"])
            if "color" in data:
                self.color = bool(data["color"])

        print(f"Model loaded from {path}")