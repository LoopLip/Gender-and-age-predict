import pytest
pytest.importorskip('numpy')
pytest.importorskip('cv2')
pytest.importorskip('pandas')

from pathlib import Path
import types
import numpy as np
import pandas as pd
import cv2

from src.generator import ImageSequence


def test_image_sequence_basic(tmp_path):
    # Create a small dataset under project_root/data/testdb_crop
    repo_root = Path(__file__).resolve().parents[2]
    img_dir = repo_root.joinpath("data", "testdb_crop")
    img_dir.mkdir(parents=True, exist_ok=True)

    # create two small images
    img1 = np.full((10, 10, 3), 255, dtype=np.uint8)
    img2 = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.imwrite(str(img_dir.joinpath("img1.jpg")), img1)
    cv2.imwrite(str(img_dir.joinpath("img2.jpg")), img2)

    df = pd.DataFrame(
        {"img_paths": ["img1.jpg", "img2.jpg"], "genders": [0, 1], "ages": [25, 30]}
    )

    cfg = types.SimpleNamespace(
        data=types.SimpleNamespace(db="testdb"),
        model=types.SimpleNamespace(img_size=64, model_name="EfficientNetB3"),
        train=types.SimpleNamespace(batch_size=2, seed=42),
    )

    gen = ImageSequence(cfg, df, "train")
    imgs, (genders, ages) = gen[0]

    assert imgs.shape == (2, 64, 64, 3)
    assert imgs.dtype == np.float32
    assert imgs.max() <= 1.0
    assert list(genders) == [0, 1]
    assert list(ages) == [25, 30]
