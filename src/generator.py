from pathlib import Path
import numpy as np
import cv2
import albumentations as A
from tensorflow.keras.utils import Sequence
from typing import Optional, Callable

# Augmentations (applied only in training mode)
_transforms = A.Compose([
    # Geometric + photometric augmentations to improve robustness
    A.Affine(translate_percent={'x': (-0.03125, 0.03125), 'y': (-0.03125, 0.03125)},
             scale=(0.8, 1.2),
             rotate=(-20, 20),
             p=0.8),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussianBlur(blur_limit=(3,7), p=0.3),
    A.MotionBlur(blur_limit=5, p=0.2),
    A.CoarseDropout(p=0.3),
    A.HorizontalFlip(p=0.5),
])


class ImageSequence(Sequence):
    """Keras Sequence for loading images and labels.

    Returns: (images, (genders, ages)) where images are float32 in [0,1].
    """

    def __init__(self, cfg, df, mode: str, preprocess_fn: Optional[Callable] = None):
        self.cfg = cfg
        self.df = df.reset_index(drop=True)
        self.indices = np.arange(len(self.df))
        self.batch_size = int(cfg.train.batch_size)
        self.img_dir = (
            Path(__file__).resolve().parents[1].joinpath("data", f"{cfg.data.db}_crop")
        )
        self.img_size = int(cfg.model.img_size)
        self.mode = mode
        self.preprocess_fn = preprocess_fn

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.df))
        sample_indices = self.indices[start:end]

        imgs = np.empty(
            (len(sample_indices), self.img_size, self.img_size, 3), dtype=np.float32
        )
        genders = np.empty((len(sample_indices),), dtype=np.int32)
        # ages as integer label (0..100) for distribution head
        ages = np.empty((len(sample_indices),), dtype=np.int32)

        for i, row_idx in enumerate(sample_indices):
            row = self.df.iloc[row_idx]
            img_path = (
                self.img_dir.joinpath(row["img_paths"])
                if "img_paths" in row
                else self.img_dir.joinpath(row["image"])
            )
            # avoid noisy OpenCV warning by checking file existence first
            if not img_path.exists():
                img = None
            else:
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                # replace unreadable or missing image with zeros
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, (self.img_size, self.img_size))

            if self.mode == "train":
                img = _transforms(image=img)["image"]

            img = img.astype(np.float32) / 255.0
            if self.preprocess_fn:
                img = self.preprocess_fn(img)

            imgs[i] = img
            genders[i] = int(row.get("genders", row.get("gender", 0)))
            ages[i] = int(row.get("ages", row.get("age", 0)))

        # prepare outputs: one-hot genders and age distributions
        batch = len(sample_indices)
        genders_oh = np.zeros((batch, 2), dtype=np.float32)
        ages_dist = np.zeros((batch, 101), dtype=np.float32)
        for i in range(batch):
            g = int(genders[i])
            genders_oh[i, g] = 1.0
            a = int(ages[i])
            a = max(0, min(100, a))
            ages_dist[i, a] = 1.0

        # gaussian smoothing of age dist
        kernel = np.array([0.05, 0.25, 0.4, 0.25, 0.05], dtype=np.float32)
        ages_dist_sm = np.stack([np.convolve(ages_dist[i], kernel, mode='same') for i in range(batch)])

        # optional CutMix-like mixing
        mix_prob = float(getattr(self.cfg.train, 'mixup_prob', 0.0))
        mix_alpha = float(getattr(self.cfg.train, 'mixup_alpha', 0.0))
        cutmix_prob = float(getattr(self.cfg.train, 'cutmix_prob', 0.0))
        mixed = False
        if self.mode == 'train' and batch > 1 and (np.random.rand() < mix_prob or np.random.rand() < cutmix_prob):
            mixed = True
            if np.random.rand() < cutmix_prob:
                # CutMix rectangle
                idxs = np.arange(batch)
                np.random.shuffle(idxs)
                for i in range(batch):
                    j = idxs[i]
                    # random box
                    rx = np.random.randint(0, self.img_size)
                    ry = np.random.randint(0, self.img_size)
                    rw = int(self.img_size * np.random.uniform(0.2, 0.5))
                    rh = int(self.img_size * np.random.uniform(0.2, 0.5))
                    x1 = max(0, rx - rw // 2)
                    y1 = max(0, ry - rh // 2)
                    x2 = min(self.img_size, x1 + rw)
                    y2 = min(self.img_size, y1 + rh)
                    imgs[i, y1:y2, x1:x2, :] = imgs[j, y1:y2, x1:x2, :]
                    # combine labels
                    genders_oh[i] = 0.5 * genders_oh[i] + 0.5 * genders_oh[j]
                    ages_dist_sm[i] = 0.5 * ages_dist_sm[i] + 0.5 * ages_dist_sm[j]
            else:
                # MixUp
                lam = np.random.beta(mix_alpha, mix_alpha) if mix_alpha > 0 else 0.5
                idxs = np.arange(batch)
                np.random.shuffle(idxs)
                imgs = imgs * lam + imgs[idxs] * (1.0 - lam)
                genders_oh = genders_oh * lam + genders_oh[idxs] * (1.0 - lam)
                ages_dist_sm = ages_dist_sm * lam + ages_dist_sm[idxs] * (1.0 - lam)

        # if not mixed, return integer labels for genders and ages to keep API simple
        if not mixed:
            genders_out = np.array([int(g) for g in np.argmax(genders_oh, axis=1)], dtype=np.int32)
            ages_out = np.array([int(np.argmax(a)) for a in ages_dist_sm], dtype=np.int32)
            return imgs, (genders_out, ages_out)

        return imgs, (genders_oh, ages_dist_sm)

    def __len__(self):
        # ensure last partial batch is used as well
        return int(np.ceil(len(self.df) / float(self.batch_size)))

    def on_epoch_end(self):
        np.random.shuffle(self.indices)
