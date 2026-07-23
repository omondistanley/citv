"""Pix2SG wrapper smoke test (legacy file name kept for CI references)."""

import numpy as np
import torch

from scene_understanding import Pix2SGWrapper


def test_pix2sg_wrapper_initializes_and_predict_empty():
    device = torch.device("cpu")
    wrapper = Pix2SGWrapper(device)
    st = wrapper.status()
    assert "active" in st and "backend" in st
    assert wrapper.is_active()

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    out = wrapper.predict(img, image_stem="", detections=None)
    assert out == []
