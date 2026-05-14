import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from tqdm import tqdm
import tensorflow as tf

from train import create_dir, load_dataset
from metrics import dice_loss, dice_coef, precision, recall, iou

IMG_H = 256
IMG_W = 256


bce = tf.keras.losses.BinaryCrossentropy()

def total_loss(y_true, y_pred):
    return bce(y_true, y_pred) + dice_loss(y_true, y_pred)


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("results")

    """ Load the model """
    model_path = os.path.join("files", "model.keras")

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "total_loss": total_loss,
            "dice_loss": dice_loss,
            "dice_coef": dice_coef,
            "iou": iou,
            "precision": precision,
            "recall": recall
        }
    )

    """ Dataset """
    dataset_path = "Kvasir-SEG"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)

    print(f"Train: \t{len(train_x)} - {len(train_y)}")
    print(f"Valid: \t{len(valid_x)} - {len(valid_y)}")
    print(f"Test: \t{len(test_x)} - {len(test_y)}")

    """ Initialize metrics """
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    """ Prediction """
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        name = x.split("/")[-1].split(".")[0]

        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (IMG_W, IMG_H))

        x_input = image / 255.0
        x_input = np.expand_dims(x_input, axis=0).astype(np.float32)

        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        mask = mask / 255.0

        pred = model.predict(x_input, verbose=0)[0]
        pred = (pred > 0.5).astype(np.float32)

        mask_bin = (mask > 0.5).astype(np.float32)

        mask_flat = mask_bin.flatten()
        pred_flat = pred[:, :, 0].flatten()

        tp = np.sum(mask_flat * pred_flat)
        fp = np.sum((1 - mask_flat) * pred_flat)
        fn = np.sum(mask_flat * (1 - pred_flat))
        tn = np.sum((1 - mask_flat) * (1 - pred_flat))

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        """ Visualization: polyp only with original colors """
        pred_mask = pred[:, :, 0]
        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

        output = cv2.bitwise_and(image, image, mask=pred_mask)

        save_image_path = os.path.join("results", f"{name}.png")
        cv2.imwrite(save_image_path, output)

    final_precision = total_tp / (total_tp + total_fp + 1e-7)
    final_recall = total_tp / (total_tp + total_fn + 1e-7)
    final_accuracy = (total_tp + total_tn) / (
        total_tp + total_fp + total_fn + total_tn + 1e-7
    )

    final_dice = (2 * total_tp) / (
        2 * total_tp + total_fp + total_fn + 1e-7
    )

    final_dice_loss = 1 - final_dice

    final_iou = total_tp / (
        total_tp + total_fp + total_fn + 1e-7
    )

    final_specificity = total_tn / (
        total_tn + total_fp + 1e-7
    )

    print("\nFinal Results:")
    print(f"Precision: {final_precision:.4f}")
    print(f"Recall/Sensitivity: {final_recall:.4f}")
    print(f"Specificity: {final_specificity:.4f}")
    print(f"Accuracy: {final_accuracy:.4f}")
    print(f"Dice Coefficient: {final_dice:.4f}")
    print(f"Dice Loss: {final_dice_loss:.4f}")
    print(f"IoU/Jaccard: {final_iou:.4f}")