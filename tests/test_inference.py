import pytest
pytest.importorskip('numpy')

import numpy as np
from src.inference import detect_faces


class FakeDet:
    def __call__(self, img, upsample):
        return []


def test_detect_faces_no_faces():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    faces, boxes = detect_faces(FakeDet(), img, margin=0.4, img_size=64)
    assert faces == []
    assert boxes == []
