from pathlib import Path
import os
import multiprocessing
import logging
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import (LearningRateScheduler, ModelCheckpoint,
                                        ReduceLROnPlateau, EarlyStopping, TensorBoard)
from src.factory import get_model, get_optimizer, get_scheduler, get_preprocess_fn
from src.generator import ImageSequence
from src.logger import get_logger
from omegaconf import OmegaConf

logger = get_logger("train")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main():
    # load config via OmegaConf for compatibility
    cfg = OmegaConf.load(Path(__file__).parent.joinpath('src', 'config.yaml'))
    set_seed(int(cfg.train.get('seed', 42)))

    # allow overriding CSV via env var (useful for local tests)
    env_csv = os.environ.get('TRAIN_CSV')
    if env_csv:
        csv_path = Path(env_csv)
    else:
        csv_path = Path(__file__).parent.joinpath("meta", f"{cfg.data.db}.csv")

    if not csv_path.exists():
        raise SystemExit(f"Meta CSV not found: {csv_path}")

    # read CSV robustly and log preview
    try:
        df = pd.read_csv(str(csv_path))
    except Exception as e:
        raise SystemExit(f"Failed to read CSV {csv_path}: {e}")
    if df is None or len(df) == 0:
        # dump file preview for debugging
        logger.error("CSV contents:\n" + "\n".join(Path(csv_path).read_text().splitlines()[:20]))
        raise SystemExit(f"Meta CSV is empty: {csv_path}")

    # Ensure at least one sample remains in train
    if len(df) < 2:
        raise SystemExit(f"Not enough samples ({len(df)}) in meta CSV to split for train/val")

    train_df, val_df = train_test_split(df, random_state=int(cfg.train.get('seed', 42)), test_size=0.1)

    # use backbone-specific preprocess if available
    preprocess_fn = get_preprocess_fn(cfg)
    if preprocess_fn is None:
        logger.info("No specific preprocess_fn found for backbone, using default 0-1 scaling")
    else:
        logger.info("Using backbone-specific preprocess_fn")

    train_gen = ImageSequence(cfg, train_df, "train", preprocess_fn=preprocess_fn)
    val_gen = ImageSequence(cfg, val_df, "val", preprocess_fn=preprocess_fn)

    # choose distribution strategy if GPUs available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        strategy = tf.distribute.MirroredStrategy()
        logger.info(f"Using MirroredStrategy with {len(gpus)} GPUs")
    else:
        strategy = tf.distribute.get_strategy()
        logger.info("No GPUs detected, using default strategy")

    with strategy.scope():
        model = get_model(cfg)
        opt = get_optimizer(cfg)
        scheduler = get_scheduler(cfg)
                # pred_gender: classification (sparse labels), pred_age: distribution (101 classes)
        # mixed age loss: alpha * MAE(expected_age, true_age) + (1-alpha) * KLD(age_dist, soft_label)
        alpha = float(cfg.train.get('alpha_age_loss', 0.5))
        # warmup schedule params
        warmup_epochs = int(cfg.train.get('warmup_epochs', 2))
        total_epochs = int(cfg.train.get('epochs', 10))

        def lr_with_warmup(epoch):
            if epoch < warmup_epochs:
                return float(cfg.train.lr) * (0.1 + 0.9 * (epoch / max(1, warmup_epochs - 1)))
            return scheduler(epoch)

        # callback to unfreeze top layers after certain epochs
        class UnfreezeCallback(tf.keras.callbacks.Callback):
            def __init__(self, unfreeze_after=2, finetune_layers=0):
                super().__init__()
                self.unfreeze_after = unfreeze_after
                self.finetune_layers = finetune_layers
                self._done = False

            def on_epoch_end(self, epoch, logs=None):
                if not self._done and epoch + 1 >= self.unfreeze_after:
                    # unfreeze last finetune_layers of the model (excluding heads)
                    if self.finetune_layers > 0:
                        layers = self.model.layers
                        # attempt to skip final heads (assume last two are heads)
                        base_layers = layers[:-2]
                        for layer in base_layers[-self.finetune_layers:]:
                            layer.trainable = True
                        self._done = True
                        print(f"Unfroze top {self.finetune_layers} base layers at epoch {epoch+1}")

        def mixed_age_loss(y_true_int, y_pred_dist):
            # y_true_int: either (batch,) int labels or (batch,101) distribution
            kernel_vals = tf.constant([0.05, 0.25, 0.4, 0.25, 0.05], dtype=tf.float32)
            kernel_1d = tf.reshape(kernel_vals, [-1, 1, 1])  # (kw, in_ch, out_ch)

            def _process_as_int(y):
                y_true = tf.cast(tf.reshape(y, [-1]), tf.int32)
                one_hot = tf.one_hot(y_true, depth=101)  # (b,101)
                inp = tf.expand_dims(one_hot, axis=2)  # (b,101,1)
                smoothed = tf.nn.conv1d(inp, kernel_1d, stride=1, padding='SAME')  # (b,101,1)
                smoothed = tf.squeeze(smoothed, axis=2)  # (b,101)
                return smoothed

            def _process_as_dist(y):
                y_float = tf.cast(y, tf.float32)
                inp = tf.expand_dims(y_float, axis=2)
                smoothed = tf.nn.conv1d(inp, kernel_1d, stride=1, padding='SAME')
                return tf.squeeze(smoothed, axis=2)

            # use tf.cond to branch in graph mode
            is_int = tf.equal(tf.rank(y_true_int), 1)
            y_true_dist = tf.cond(is_int, lambda: _process_as_int(y_true_int), lambda: _process_as_dist(y_true_int))

            # KLD between smoothed true distribution and predicted distribution
            kld = tf.keras.losses.KLDivergence()(y_true_dist, y_pred_dist)
            # expected ages
            ages = tf.range(0, 101, dtype=tf.float32)
            expected_pred = tf.reduce_sum(y_pred_dist * ages[None, :], axis=1)
            expected_true = tf.reduce_sum(y_true_dist * ages[None, :], axis=1)
            mae = tf.keras.losses.MeanAbsoluteError()(expected_true, expected_pred)
            return alpha * mae + (1.0 - alpha) * kld

        # compile model with mixed loss for age
        # define metrics: categorical_accuracy for gender and KLD+expected_mae for age
        kld_loss = tf.keras.losses.KLDivergence()
        def kld_metric(y_true, y_pred):
            return kld_loss(y_true, y_pred)
        def expected_age_mae(y_true, y_pred):
            ages = tf.range(0, 101, dtype=tf.float32)
            true_exp = tf.reduce_sum(tf.cast(y_true, tf.float32) * ages[None, :], axis=1)
            pred_exp = tf.reduce_sum(y_pred * ages[None, :], axis=1)
            return tf.reduce_mean(tf.abs(true_exp - pred_exp))

        model.compile(
            optimizer=opt,
            loss=["categorical_crossentropy", mixed_age_loss],
            metrics=[ ["categorical_accuracy"], [kld_metric, expected_age_mae] ],
        )

    checkpoint_dir = Path(__file__).parent.joinpath("checkpoint")
    checkpoint_dir.mkdir(exist_ok=True)
    # Keras 3 requires .keras extension for model.save; include format in filename
    filename = "_".join([cfg.model.model_name,
                          str(cfg.model.img_size),
                          "weights.{epoch:02d}-{val_loss:.2f}.keras"])

    callbacks = [
        LearningRateScheduler(schedule=lr_with_warmup),
        ModelCheckpoint(str(checkpoint_dir) + "/" + filename,
                        monitor="val_loss",
                        verbose=1,
                        save_best_only=True,
                        mode="auto"),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True),
        TensorBoard(log_dir=str(checkpoint_dir.joinpath('logs')))
    ]

    # unfreeze callback if requested
    unfreeze_after = int(cfg.train.get('unfreeze_after', 2))
    finetune_layers = int(cfg.train.get('finetune_layers', 0))
    if finetune_layers > 0:
        callbacks.append(UnfreezeCallback(unfreeze_after=unfreeze_after, finetune_layers=finetune_layers))

    # CSV logger for epoch results
    csv_log = checkpoint_dir.joinpath('training_log.csv')
    class SimpleCSVLogger(tf.keras.callbacks.Callback):
        def __init__(self, path):
            super().__init__()
            self.path = path
            if not self.path.exists():
                self.path.write_text('epoch,' + ','.join(['loss','val_loss','val_pred_age_kullback_leibler_divergence','val_expected_age_mae']) + '\n')

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            line = f"{epoch+1},{logs.get('loss', '')},{logs.get('val_loss', '')},{logs.get('val_pred_age_kullback_leibler_divergence', '')},{logs.get('val_pred_age_mae', '')}\n"
            self.path.write_text(self.path.read_text() + line)
    callbacks.append(SimpleCSVLogger(csv_log))

    # integrate wandb if configured
    if cfg.wandb.project:
        import wandb
        from wandb.keras import WandbCallback
        wandb.init(project=cfg.wandb.project)
        callbacks.append(WandbCallback())

    steps_per_epoch = int(np.ceil(len(train_df) / cfg.train.batch_size))
    validation_steps = int(np.ceil(len(val_df) / cfg.train.batch_size))

    model.fit(
        train_gen,
        epochs=cfg.train.epochs,
        callbacks=callbacks,
        validation_data=val_gen,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )


if __name__ == '__main__':
    main()
