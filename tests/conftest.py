import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def tiny_image_dataset(tmp_path):
    root = tmp_path / "raw-img"
    classes = ["cane", "gatto"]

    for class_name in classes:
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(4):
            rng = np.random.default_rng(seed=index)
            image_array = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
            image = Image.fromarray(image_array, mode="RGB")
            image.save(class_dir / f"sample_{index}.jpg")

    return root


@pytest.fixture(autouse=True)
def deterministic_seed():
    random.seed(42)
    np.random.seed(42)
