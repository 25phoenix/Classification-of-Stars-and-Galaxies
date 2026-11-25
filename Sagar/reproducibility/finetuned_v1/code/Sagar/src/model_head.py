from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from .config import ALL_CLASS_IDS, HeadHyperparams, NUM_CLASSES


@dataclass
class FoldTrainResult:
    model: tf.keras.Model
    val_probs: np.ndarray
    best_val_macro_f1: float
    best_epoch: int
    history: dict[str, list[float]]
    train_seconds: float


def compute_fold_class_weights(y_train: np.ndarray) -> dict[int, float]:
    """Compute balanced class weights strictly from the current training partition."""
    present = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=present, y=y_train)
    return {int(c): float(w) for c, w in zip(present, weights)}


def build_classification_head(
    embedding_dim: int,
    hparams: HeadHyperparams,
    seed: int = 42,
) -> tf.keras.Model:
    """Build a compact 10-class classification head on frozen MobileNetV2 embeddings."""
    tf.keras.utils.set_random_seed(seed)
    reg = tf.keras.regularizers.l2(hparams.l2_reg) if hparams.l2_reg > 0 else None

    inputs = tf.keras.Input(shape=(embedding_dim,), name="mobilenetv2_embedding")
    x = inputs
    if hparams.dropout > 0:
        x = tf.keras.layers.Dropout(hparams.dropout, name="head_dropout_1")(x)
    if hparams.hidden_units > 0:
        x = tf.keras.layers.Dense(
            hparams.hidden_units,
            activation="relu",
            kernel_regularizer=reg,
            name="head_dense",
        )(x)
        if hparams.dropout > 0:
            x = tf.keras.layers.Dropout(hparams.dropout, name="head_dropout_2")(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        kernel_regularizer=reg,
        dtype="float32",
        name="galaxy10_probs",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_galaxy10_head")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hparams.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class _ValMacroF1EarlyStopping(tf.keras.callbacks.Callback):
    """Track validation macro-F1 over all 10 classes and restore best weights."""

    def __init__(
        self,
        x_val: np.ndarray,
        y_val: np.ndarray,
        patience: int = 6,
        min_delta: float = 1e-4,
    ) -> None:
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.patience = patience
        self.min_delta = min_delta
        self.best_f1 = -1.0
        self.best_epoch = 1
        self.best_weights: list[np.ndarray] | None = None
        self.wait = 0
        self.val_macro_f1_history: list[float] = []

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        probs = self.model.predict(self.x_val, batch_size=256, verbose=0)
        preds = np.argmax(probs, axis=1)
        macro_f1 = float(
            f1_score(
                self.y_val,
                preds,
                labels=ALL_CLASS_IDS,
                average="macro",
                zero_division=0,
            )
        )
        self.val_macro_f1_history.append(macro_f1)
        if logs is not None:
            logs["val_macro_f1"] = macro_f1

        if macro_f1 > self.best_f1 + self.min_delta:
            self.best_f1 = macro_f1
            self.best_epoch = epoch + 1
            self.best_weights = [w.copy() for w in self.model.get_weights()]
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def train_head_with_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    hparams: HeadHyperparams,
    max_epochs: int = 40,
    patience: int = 6,
    batch_size: int = 64,
    seed: int = 42,
) -> FoldTrainResult:
    """Train head on (x_train, y_train) with early stopping on val macro-F1."""
    model = build_classification_head(x_train.shape[1], hparams, seed=seed)
    class_weight = compute_fold_class_weights(y_train) if hparams.use_class_weight else None
    cb = _ValMacroF1EarlyStopping(x_val=x_val, y_val=y_val, patience=patience)

    t0 = time.perf_counter()
    hist = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=max_epochs,
        batch_size=batch_size,
        shuffle=True,
        class_weight=class_weight,
        callbacks=[cb],
        verbose=0,
    )
    train_seconds = time.perf_counter() - t0

    val_probs = model.predict(x_val, batch_size=256, verbose=0).astype(np.float32)
    history_dict = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    history_dict["val_macro_f1"] = cb.val_macro_f1_history

    return FoldTrainResult(
        model=model,
        val_probs=val_probs,
        best_val_macro_f1=cb.best_f1,
        best_epoch=cb.best_epoch,
        history=history_dict,
        train_seconds=float(train_seconds),
    )


def select_refit_epochs(fold_best_epochs: list[int]) -> int:
    """
    Development-only refit epoch rule:
    take the ceiling of the median early-stopped best epoch across tuning folds.
    """
    if not fold_best_epochs:
        return 1
    return max(1, int(np.ceil(float(np.median(fold_best_epochs)))))


def refit_head_fixed_epochs(
    x_train: np.ndarray,
    y_train: np.ndarray,
    hparams: HeadHyperparams,
    epochs: int,
    batch_size: int = 64,
    seed: int = 42,
) -> tuple[tf.keras.Model, dict[str, list[float]], float]:
    """Refit head on a full training partition for an already-selected epoch count."""
    model = build_classification_head(x_train.shape[1], hparams, seed=seed)
    class_weight = compute_fold_class_weights(y_train) if hparams.use_class_weight else None

    t0 = time.perf_counter()
    hist = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        class_weight=class_weight,
        verbose=0,
    )
    train_seconds = time.perf_counter() - t0
    history_dict = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    return model, history_dict, float(train_seconds)