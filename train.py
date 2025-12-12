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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-demo', action='store_true')
    args = parser.parse_args()

    # create essential dirs
    root = Path(__file__).parent
    (root / 'data').mkdir(exist_ok=True)
    (root / 'meta').mkdir(exist_ok=True)
    (root / 'checkpoint').mkdir(exist_ok=True)
    (root / 'pretrained_models').mkdir(exist_ok=True)

    # load config via OmegaConf for compatibility
    cfg = OmegaConf.load(Path(__file__).parent.joinpath('src', 'config.yaml'))
    # allow CLI flag to trigger demo generation
    if args.generate_demo:
        import os
        os.environ['GENERATE_DEMO'] = '1'
    set_seed(int(cfg.train.get('seed', 42)))

    # allow overriding CSV via env var (useful for local tests)
    env_csv = os.environ.get('TRAIN_CSV')
    if env_csv:
        csv_path = Path(env_csv)
    else:
        csv_path = Path(__file__).parent.joinpath("meta", f"{cfg.data.db}.csv")

    # support CLI flag --generate-demo via env var for backward compatibility
    gen_demo = os.environ.get('GENERATE_DEMO', '0') == '1'

    if not csv_path.exists() and not gen_demo:
        logger.warning(f"Meta CSV not found: {csv_path}. Please provide dataset (IMDB-WIKI/UTK) or run with --generate-demo to create a tiny demo for tests. Training will abort until dataset available.")
        raise SystemExit(1)

    if not csv_path.exists() and gen_demo:
        logger.info(f"Creating a mini demo dataset for testing at {csv_path}.")
        # create meta and data dirs
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        demo_img_dir = Path(__file__).parent.joinpath("data", f"{cfg.data.db}_crop")
        demo_img_dir.mkdir(parents=True, exist_ok=True)
        # create two demo images if missing
        try:
            import cv2
            import numpy as _np
            for name, age, gender in [("exampleOne.jpg", 25, 0), ("exampleTwo.jpg", 30, 1)]:
                p = demo_img_dir.joinpath(name)
                if not p.exists():
                    img = _np.full((int(cfg.model.img_size), int(cfg.model.img_size), 3), 128, dtype=_np.uint8)
                    cv2.putText(img, name.split('.')[0], (5, int(cfg.model.img_size)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    cv2.imwrite(str(p), img)
        except Exception:
            logger.exception("Failed to generate demo images (cv2 may be missing). Creating placeholder files instead.")
            for name, _, _ in [("exampleOne.jpg",25,0),("exampleTwo.jpg",30,1)]:
                p = demo_img_dir.joinpath(name)
                if not p.exists():
                    p.write_bytes(b'')
        # create demo CSV
        demo_df = pd.DataFrame([
            {"image": "exampleOne.jpg", "gender": 0, "age": 25},
            {"image": "exampleTwo.jpg", "gender": 1, "age": 30},
        ])
        demo_df.to_csv(str(csv_path), index=False)
        logger.info(f"Created demo meta at {csv_path} and images in {demo_img_dir}")

    # read CSV robustly and log preview
    try:
        df = pd.read_csv(str(csv_path))
    except Exception as e:
        raise SystemExit(f"Failed to read CSV {csv_path}: {e}")
    if df is None or len(df) == 0:
        # dump file preview for debugging
        logger.error("CSV contents:\n" + "\n".join(Path(csv_path).read_text().splitlines()[:20]))
        raise SystemExit(f"Meta CSV is empty: {csv_path}")
    # show preview and check first N image paths
    try:
        preview = df.head(10)
        logger.info("Meta CSV preview:\n" + preview.to_string())
        # check if listed image files exist (relative to data/{db}_crop)
        sample_img_dir = Path(__file__).parent.joinpath('data', f"{cfg.data.db}_crop")
        missing = []
        for _, row in preview.iterrows():
            img_name = row.get('img_paths') if 'img_paths' in row else row.get('image')
            if pd.isna(img_name):
                continue
            path = sample_img_dir.joinpath(str(img_name))
            if not path.exists():
                missing.append(str(path))
        if missing:
            logger.warning(f"{len(missing)} of first {len(preview)} images missing. Example missing: {missing[:3]}")
    except Exception:
        logger.exception("Failed to preview/check meta CSV")

    # Ensure at least one sample remains in train
    if len(df) < 2:
        raise SystemExit(f"Not enough samples ({len(df)}) in meta CSV to split for train/val")

    train_df, val_df = train_test_split(df, random_state=int(cfg.train.get('seed', 42)), test_size=0.1)

    # use backbone-specific preprocess if available and auto-download weights if requested
    preprocess_fn = get_preprocess_fn(cfg)
    if preprocess_fn is None:
        logger.info("No specific preprocess_fn found for backbone, using default 0-1 scaling")
    else:
        logger.info("Using backbone-specific preprocess_fn")

    # if model requires pretrained weights, ensure Keras downloads them automatically via get_model (weights='imagenet')
    # create data generators
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
            # Normalize any incoming label format to a distribution (b,101): supports (b,), (b,1), (b,101)
            kernel_vals = tf.constant([0.05, 0.25, 0.4, 0.25, 0.05], dtype=tf.float32)
            kernel_1d = tf.reshape(kernel_vals, [-1, 1, 1])  # (kw, in_ch, out_ch)

            y = y_true_int
            y_rank = tf.rank(y)
            # if shape (b,): convert to int and one-hot
            def _from_1d():
                y0 = tf.cast(tf.reshape(y, [-1]), tf.int32)
                return tf.one_hot(y0, depth=101)
            # if shape (b,1): squeeze then one-hot if ints, or if floats assume distribution
            def _from_2d():
                s = tf.shape(y)
                second = tf.gather(s, 1)
                y2 = tf.reshape(y, [-1, second])
                # if second==1 treat as ints
                def _as_int2():
                    vals = tf.cast(tf.reshape(y2, [-1]), tf.int32)
                    return tf.one_hot(vals, depth=101)
                def _as_dist2():
                    return tf.cast(y2, tf.float32)
                return tf.cond(tf.equal(second, 1), _as_int2, _as_dist2)
            y_dist = tf.cond(tf.equal(y_rank, 1), _from_1d, _from_2d)

            # ensure float and add channel for conv1d via conv1d (expects [b, length, channels])
            y_float = tf.cast(y_dist, tf.float32)
            inp = tf.expand_dims(y_float, axis=2)  # (b,101,1)
            smoothed = tf.nn.conv1d(inp, kernel_1d, stride=1, padding='SAME')  # (b,101,1)
            y_true_dist = tf.squeeze(smoothed, axis=2)  # (b,101)

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
        def _y_to_dist(y):
            y = tf.convert_to_tensor(y)
            rank = tf.rank(y)
            def _from_1d():
                y0 = tf.cast(tf.reshape(y, [-1]), tf.int32)
                return tf.one_hot(y0, depth=101)
            def _from_2d():
                s = tf.shape(y)
                second = tf.gather(s, 1)
                y2 = tf.reshape(y, [-1, second])
                def _as_int2():
                    vals = tf.cast(tf.reshape(y2, [-1]), tf.int32)
                    return tf.one_hot(vals, depth=101)
                def _as_dist2():
                    return tf.cast(y2, tf.float32)
                return tf.cond(tf.equal(second, 1), _as_int2, _as_dist2)
            return tf.cond(tf.equal(rank, 1), _from_1d, _from_2d)

        def kld_metric(y_true, y_pred):
            y_true_dist = _y_to_dist(y_true)
            return kld_loss(y_true_dist, y_pred)
        def expected_age_mae(y_true, y_pred):
            y_true_dist = _y_to_dist(y_true)
            ages = tf.range(0, 101, dtype=tf.float32)
            true_exp = tf.reduce_sum(tf.cast(y_true_dist, tf.float32) * ages[None, :], axis=1)
            pred_exp = tf.reduce_sum(y_pred * ages[None, :], axis=1)
            return tf.reduce_mean(tf.abs(true_exp - pred_exp))

        model.compile(
            optimizer=opt,
            loss=["sparse_categorical_crossentropy", mixed_age_loss],
            metrics=[ ["sparse_categorical_accuracy"], [kld_metric, expected_age_mae] ],
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

    # callback to log sample predictions after each epoch
    class SamplePredictionsCallback(tf.keras.callbacks.Callback):
        def __init__(self, val_df, preprocess_fn, sample_count=3):
            super().__init__()
            self.val_df = val_df.reset_index(drop=True)
            self.img_dir = Path(__file__).parent.joinpath('data', f"{cfg.data.db}_crop")
            self.sample_count = sample_count
            self.preprocess_fn = preprocess_fn
            # choose first few available samples
            self.samples = []
            for i in range(min(len(self.val_df), 50)):
                row = self.val_df.iloc[i]
                img_name = row.get('img_paths') if 'img_paths' in row else row.get('image')
                if pd.isna(img_name):
                    continue
                p = self.img_dir.joinpath(str(img_name))
                if p.exists():
                    self.samples.append((p, int(row.get('age', row.get('ages', 0))), int(row.get('gender', row.get('genders', 0)))))
                if len(self.samples) >= self.sample_count:
                    break

        def on_epoch_end(self, epoch, logs=None):
            import cv2
            import numpy as _np
            if not hasattr(self.model, 'predict'):
                return
            print('\nSample predictions after epoch', epoch+1)
            for p, true_age, true_gender in self.samples:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                img = cv2.resize(img, (int(cfg.model.img_size), int(cfg.model.img_size)))
                img = img.astype(_np.float32) / 255.0
                if self.preprocess_fn:
                    img = self.preprocess_fn(img)
                inp = _np.expand_dims(img, axis=0)
                preds = self.model.predict(inp)
                # preds: [gender_prob, age_dist]
                gender_prob = preds[0][0] if isinstance(preds, list) else preds[0]
                age_dist = preds[1][0] if isinstance(preds, list) else preds[1]
                pred_age = int(_np.round(_np.sum(_np.arange(0,101) * age_dist)))
                top5_idx = _np.argsort(age_dist)[-5:][::-1]
                top5 = [(int(i), float(age_dist[i])) for i in top5_idx]
                print(f"Image: {p.name} True age: {true_age} Pred age: {pred_age} Top-5 age probs: {top5}")
    callbacks.append(SamplePredictionsCallback(val_df, preprocess_fn))

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
