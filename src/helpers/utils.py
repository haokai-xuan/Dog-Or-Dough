import cv2
import numpy as np
import os

def preprocess_image(img_path, target_size=(64, 64)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    img = img.transpose(2, 0, 1)
    return img.reshape(-1, 1)

def load_dataset(data_dir, target_size=(64, 64)):
    X_list, Y_list = [], []
    classes = sorted(os.listdir(data_dir))

    for i, cls in enumerate(classes):
        class_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            img_path = os.path.join(class_dir, fname)
            X_list.append(preprocess_image(img_path, target_size))
            Y_list.append(i)


    X = np.hstack(X_list)
    Y = np.array(Y_list, dtype=int)
    Y = np.eye(len(classes))[:, Y]

    return X, Y

def shuffle(X, Y):
    perm = np.random.permutation(X.shape[1])
    return X[:, perm], Y[:, perm]

def get_batches(X, Y, batch_size):
    m = X.shape[1]
    X, Y = shuffle(X, Y)
    batches = []

    for start in range(0, m, batch_size):
        end = start + batch_size
        X_batch  = X[:, start:end]
        Y_batch  = Y[:, start:end]
        batches.append((X_batch, Y_batch))

    return batches
