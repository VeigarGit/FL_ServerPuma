"""Opt-in real training over two cameras and two rounds, using pretrained YOLO11n."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from ultralytics import YOLO

from conftest import ROOT


@pytest.mark.skipif(os.environ.get('PUMA_YOLO_SMOKE') != '1', reason='Set PUMA_YOLO_SMOKE=1 for actual training')
def test_two_cameras_two_rounds(dataset, tmp_path):
    command = [sys.executable, str(ROOT / 'src/system/yolo/run.py'),
               '--config', str(ROOT / 'configs/yolo.yaml'), '--dataset-root', str(dataset),
               '--cameras', 'camera_01', 'camera_02', '--rounds', '2',
               '--device', 'cpu', '--imgsz', '64', '--batch', '2',
               '--output', str(tmp_path / 'results')]
    subprocess.run(command, cwd=ROOT, check=True, timeout=300)
    output = next((tmp_path / 'results').iterdir())
    metrics = json.loads((output / 'metrics.json').read_text())
    assert len(metrics) == 2
    for record in metrics:
        assert len(record['clients']) == 2
        assert {client['start_digest'] for client in record['clients']} == {record['global_start_digest']}
        assert record['sent_bytes'] > 0 and record['received_bytes'] > 0
        assert all(np.isfinite(record['global'][key]) for key in ('map50', 'map50_95', 'precision', 'recall'))
    assert metrics[1]['global_start_digest'] == metrics[0]['global_end_digest']
    assert metrics[0]['global_start_digest'] != metrics[0]['global_end_digest']
    assert (output / 'test.json').is_file()
    reloaded = YOLO(str(output / 'best_global.pt'))
    assert len(reloaded.names) == 8
    assert len(reloaded.predict(np.zeros((64, 64, 3), dtype=np.uint8), imgsz=64, device='cpu', verbose=False)) == 1
