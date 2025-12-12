"""Diagnostic script to inspect model outputs on sample images.
Usage:
  python tools/diagnose_model.py [--weights path/to/weights] [--n 3]

It loads latest checkpoint or provided weights, runs predictions on several images from data/{db}_crop,
computes expected age and prints top-5 age probabilities.
"""
from pathlib import Path
import sys
import os
# ensure repo root is on sys.path so 'src' can be imported
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import traceback
import argparse
import numpy as np
from omegaconf import OmegaConf

try:
    from src.factory import get_model, get_preprocess_fn
except Exception:
    print('Failed to import project modules; run from repo root.')
    raise


def find_latest_model(checkpoint_dir: Path):
    if not checkpoint_dir.exists():
        return None
    exts = ('.keras', '.h5', '.hdf5', '.ckpt', '.pt', '.pth')
    files = [p for p in checkpoint_dir.glob('*') if p.suffix.lower() in exts]
    if not files:
        return None
    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    return str(files[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--n', type=int, default=3)
    args = parser.parse_args()

    cfg = OmegaConf.load(Path('src/config.yaml'))
    db = cfg.data.db
    img_size = int(cfg.model.img_size)

    # determine weights
    weight_file = None
    if args.weights:
        weight_file = args.weights
    else:
        wp = Path('checkpoint')
        latest = find_latest_model(wp)
        if latest:
            weight_file = latest
        else:
            local = Path('pretrained_models') / f"{cfg.model.model_name}_{cfg.model.img_size}_weights.11-3.44.hdf5"
            if local.exists():
                weight_file = str(local)

    print('Using weights:', weight_file)

    model = get_model(cfg)
    if weight_file:
        try:
            model.load_weights(weight_file)
        except Exception:
            print('Failed to load weights:', weight_file)
            traceback.print_exc()
            # continue without weights

    preprocess_fn = get_preprocess_fn(cfg)

    data_dir = Path('data') / f"{db}_crop"
    if not data_dir.exists():
        print('No data found at', data_dir)
        return

    images = [p for p in data_dir.glob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    if not images:
        print('No images in', data_dir)
        return

    images = images[:args.n]
    ages_arr = np.arange(0, 101, dtype=np.float32)

    for p in images:
        try:
            import cv2
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            h,w = img.shape[:2]
            img = cv2.resize(img, (img_size, img_size))
            img = img.astype(np.float32)/255.0
            if preprocess_fn:
                try:
                    img = preprocess_fn(img)
                except Exception:
                    print('preprocess_fn failed for', p)
            x = np.expand_dims(img, 0)
            preds = model.predict(x)
            preds_age = preds[1][0]
            expected = float(np.sum(preds_age * ages_arr))
            top5 = sorted(list(enumerate(preds_age)), key=lambda x: -x[1])[:5]
            print(f'Image: {p.name} | expected_age={expected:.2f}')
            print('Top5 ages:', [(int(a), float(prob)) for a, prob in top5])
        except Exception:
            print('Failed processing', p)
            traceback.print_exc()

if __name__ == '__main__':
    main()
