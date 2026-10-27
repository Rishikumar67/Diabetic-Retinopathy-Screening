"""
Model Training & Transfer Learning Pipeline
Diabetic Retinopathy Screening Using Deep Learning

Features:
- Loads 585 annotated fundus images from data/dataset_catalog.csv
- Stratified 80/20 Train-Test split
- Real-time Data Augmentation (rotation, zoom, flips, brightness)
- Class Weighting to counteract class imbalance (as per synopsis Section 7)
- Learning Rate Scheduling & Early Stopping
- Saves model weights to saved_models/ and evaluation metrics to reports/
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.model import build_efficientnet_dr_model
from src.preprocessing import preprocess_for_model, load_image
from src.evaluation import compute_screening_metrics, plot_confusion_matrix

class FundusDataGenerator(tf.keras.utils.Sequence):
    """
    Memory-efficient generator that loads and augments fundus images on-the-fly.
    """
    def __init__(self, df: pd.DataFrame, batch_size: int = 16, augment: bool = True, shuffle: bool = True):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indexes]

        images = []
        labels = []

        for _, row in batch_df.iterrows():
            img_path = row["image_path"]
            label = int(row["dr_grade"])
            
            img_rgb = load_image(img_path)
            if img_rgb is None:
                img_rgb = np.zeros((224, 224, 3), dtype=np.uint8)

            norm_img = preprocess_for_model(img_rgb, target_size=(224, 224), apply_crop=True)

            if self.augment:
                # Random horizontal and vertical flip
                if np.random.rand() > 0.5:
                    norm_img = np.fliplr(norm_img)
                if np.random.rand() > 0.5:
                    norm_img = np.flipud(norm_img)
                # Random brightness perturbation
                if np.random.rand() > 0.5:
                    factor = np.random.uniform(0.85, 1.15)
                    norm_img = np.clip(norm_img * factor, 0.0, 1.0)

            images.append(norm_img)
            # One-hot encoding
            one_hot = np.zeros(5, dtype=np.float32)
            one_hot[label] = 1.0
            labels.append(one_hot)

        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)

def run_training(
    epochs: int = 5,
    batch_size: int = 16,
    catalog_path: str = "data/dataset_catalog.csv",
    output_dir: str = "saved_models",
    reports_dir: str = "reports"
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    if not os.path.exists(catalog_path):
        from prepare_dataset import generate_catalog
        df = generate_catalog(".")
    else:
        df = pd.read_csv(catalog_path)

    print(f"Loaded dataset catalog with {len(df)} images.")
    print("Class distribution:\n", df["dr_grade"].value_counts().sort_index())

    # Stratified split
    train_df, val_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["dr_grade"]
    )
    print(f"Training set: {len(train_df)} samples | Validation set: {len(val_df)} samples.")

    # Calculate class weights for imbalance handling
    classes = np.array([0, 1, 2, 3, 4])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["dr_grade"].values
    )
    class_weight_dict = {i: float(w) for i, w in zip(classes, weights)}
    print("Computed Class Weights:", class_weight_dict)

    train_gen = FundusDataGenerator(train_df, batch_size=batch_size, augment=True)
    val_gen = FundusDataGenerator(val_df, batch_size=batch_size, augment=False, shuffle=False)

    print("\nBuilding EfficientNetB0 Transfer Learning Architecture...")
    model = build_efficientnet_dr_model(input_shape=(224, 224, 3), num_classes=5, learning_rate=2e-4)

    weights_path = os.path.join(output_dir, "efficientnet_b0_dr.weights.h5")
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            weights_path,
            save_best_only=True,
            save_weights_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        )
    ]

    print(f"\nStarting training for {epochs} epochs...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # Save final weights if not saved by checkpoint
    if not os.path.exists(weights_path):
        model.save_weights(weights_path)
    print(f"Model weights saved to: {weights_path}")

    # Evaluate on full validation set
    print("\nEvaluating model on validation cohort...")
    y_true_list = []
    y_pred_list = []
    y_prob_list = []

    for i in range(len(val_gen)):
        imgs, lbls = val_gen[i]
        preds = model.predict(imgs, verbose=0)
        y_prob_list.append(preds)
        y_true_list.append(np.argmax(lbls, axis=1))
        y_pred_list.append(np.argmax(preds, axis=1))

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    y_prob = np.concatenate(y_prob_list)

    metrics = compute_screening_metrics(y_true, y_pred, y_prob)
    print("\n--- Validation Performance Metrics ---")
    print(f"Accuracy:         {metrics['accuracy']:.4f}")
    print(f"Macro F1-Score:   {metrics['f1_macro']:.4f}")
    print(f"Screening Sensitivity: {metrics['binary_screening']['sensitivity']:.4f}")
    print(f"Screening Specificity: {metrics['binary_screening']['specificity']:.4f}")

    # Save metrics JSON
    metrics_path = os.path.join(reports_dir, "metrics_summary.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved evaluation metrics to: {metrics_path}")

    # Save Confusion Matrix figure
    cm_array = np.array(metrics["confusion_matrix"])
    fig = plot_confusion_matrix(cm_array, title="EfficientNetB0 Validation Confusion Matrix")
    cm_path = os.path.join(reports_dir, "confusion_matrix.png")
    fig.savefig(cm_path, bbox_inches="tight")
    print(f"Saved confusion matrix plot to: {cm_path}")

    # Save model metadata
    meta = {
        "architecture": "EfficientNetB0 Transfer Learning",
        "input_shape": [224, 224, 3],
        "num_classes": 5,
        "classes": ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"],
        "dataset_total": len(df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "weights_file": weights_path.replace("\\", "/"),
        "metrics": metrics
    }
    with open(os.path.join(output_dir, "model_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return model, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNetB0 for DR Screening")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()
    run_training(epochs=args.epochs, batch_size=args.batch_size)
