import cv2
import numpy as np
import os
import albumentations as A

def preprocess_image(img_path, target_size=(64, 64)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    img = img.transpose(2, 0, 1)
    return img.reshape(-1, 1)

def load_dataset(data_dir, target_size=(64, 64)):
    X_list, Y_list = [], []
    classes = sorted(os.listdir(data_dir))
    print(classes)

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

def get_image_paths(data_dir):
    image_paths = []
    Y_list = []
    classes = sorted(os.listdir(data_dir))

    for i, cls in enumerate(classes):
        class_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            image_paths.append(os.path.join(class_dir, fname))
            Y_list.append(i)

    Y = np.array(Y_list, dtype=int)
    Y = np.eye(len(classes))[:, Y]
    return image_paths, Y

def shuffle(X, Y):
    perm = np.random.permutation(X.shape[1])
    return X[:, perm], Y[:, perm]

def get_augmentation_pipeline():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ])

def get_batches(image_paths, Y, batch_size, aug_pipeline=None, target_size=(64, 64)):
    m = len(image_paths)
    batches = []
    X_list = []

    for i in range(m):
        img = cv2.imread(image_paths[i])

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if aug_pipeline is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            augmented = aug_pipeline(image=img_rgb)
            img = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
        
        img = cv2.resize(img, target_size)
        img = img / 255.0
        img = img.transpose(2, 0, 1)
        X_list.append(img.reshape(-1, 1))

    X = np.hstack(X_list)
    X, Y = shuffle(X, Y)

    for start in range(0, m, batch_size):
        end = start + batch_size
        X_batch = X[:, start:end]
        Y_batch = Y[:, start:end]
        batches.append((X_batch, Y_batch))

    return batches
