import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import albumentations as A

from model import unet3plus
from metrics import dice_loss, dice_coef, precision, recall, iou

IMG_H = 256
IMG_W = 256


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_dataset(path, split=0.1):
    """Loading images and masks"""
    X = sorted(glob(os.path.join(path, "images", "*")))
    Y = sorted(glob(os.path.join(path, "masks", "*")))

    split_size = max(1, int(len(X) * split))

    train_x, valid_x, train_y, valid_y = train_test_split(
        X, Y,
        test_size=split_size,
        random_state=42
    )

    train_x, test_x, train_y, test_y = train_test_split(
        train_x, train_y,
        test_size=split_size,
        random_state=42
    )

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def get_train_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        A.Rotate(limit=30, p=0.5),

        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),

        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),

        A.ElasticTransform(
            alpha=1,
            sigma=50,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.3
        )
    ])


def read_image_and_mask(image_path, mask_path, augment=False):
    image_path = image_path.decode()
    mask_path = mask_path.decode()

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (IMG_W, IMG_H))

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)

    if augment:
        transform = get_train_augmentation()
        augmented = transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

    image = image / 255.0
    image = image.astype(np.float32)

    mask = mask / 255.0
    mask = mask.astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)

    return image, mask


def tf_parse_train(x, y):
    def _parse(x, y):
        return read_image_and_mask(x, y, augment=True)

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])

    x.set_shape([IMG_H, IMG_W, 3])
    y.set_shape([IMG_H, IMG_W, 1])

    return x, y


def tf_parse_valid(x, y):
    def _parse(x, y):
        return read_image_and_mask(x, y, augment=False)

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])

    x.set_shape([IMG_H, IMG_W, 3])
    y.set_shape([IMG_H, IMG_W, 1])

    return x, y


def tf_dataset(X, Y, batch=2, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))

    if augment:
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.map(tf_parse_train, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(tf_parse_valid, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


bce = tf.keras.losses.BinaryCrossentropy()

def total_loss(y_true, y_pred):
    return bce(y_true, y_pred) + dice_loss(y_true, y_pred)


if __name__ == "__main__":
    """Seeding"""
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    """Directory for storing files"""
    create_dir("files")

    """Hyperparameters"""
    batch_size = 4
    lr = 1e-4
    num_epochs = 25
    fine_tune_epochs = 10

    model_path = os.path.join("files", "model.keras")
    csv_path = os.path.join("files", "log.csv")

    """Dataset"""
    dataset_path = "Kvasir-SEG"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)

    print(f"Train: \t{len(train_x)} - {len(train_y)}")
    print(f"Valid: \t{len(valid_x)} - {len(valid_y)}")
    print(f"Test: \t{len(test_x)} - {len(test_y)}")

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size, augment=True)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size, augment=False)

    """Model"""
    model = unet3plus((IMG_H, IMG_W, 3))

    model.compile(
        loss=total_loss,
        optimizer=Adam(learning_rate=lr),
        metrics=[dice_coef, iou, precision, recall]
    )

    model.summary()

    callbacks = [
        ModelCheckpoint(
            model_path,
            monitor="val_dice_coef",
            mode="max",
            verbose=1,
            save_best_only=True
        ),

        ReduceLROnPlateau(
            monitor="val_dice_coef",
            mode="max",
            factor=0.1,
            patience=5,
            min_lr=1e-10,
            verbose=1
        ),

        CSVLogger(csv_path),

        EarlyStopping(
            monitor="val_dice_coef",
            mode="max",
            patience=15,
            restore_best_weights=True
        )
    ]

    print("\nTraining decoder with frozen encoder...")

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks
    )

    print("\nStarting fine-tuning...")

    for layer in model.layers:
        layer.trainable = True

    model.compile(
        loss=total_loss,
        optimizer=Adam(learning_rate=1e-5),
        metrics=[dice_coef, iou, precision, recall]
    )

    model.fit(
        train_dataset,
        epochs=fine_tune_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks
    )