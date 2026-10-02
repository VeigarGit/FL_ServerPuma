from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/system'))
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.yolo-settings'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.yolo-settings/matplotlib'))
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
from PIL import Image
import pytest

NAMES = dict(enumerate(['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck']))


@pytest.fixture
def names():
    return NAMES.copy()


def make_dataset(root):
    random = np.random.default_rng(42)
    for camera in ('camera_01', 'camera_02'):
        for split in ('train', 'val', 'test'):
            images = root / camera / 'images' / split
            labels = root / camera / 'labels' / split
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            for index in range(2):
                Image.fromarray(random.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(images / f'{index}.png')
                (labels / f'{index}.txt').write_text('0 0.5 0.5 0.5 0.5\n')
    return root


@pytest.fixture
def dataset(tmp_path):
    return make_dataset(tmp_path / 'dataset')
