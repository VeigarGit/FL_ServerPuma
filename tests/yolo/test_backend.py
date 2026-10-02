import torch
import pytest

from yolo.backend import FederatedTrainer, initialize, save_checkpoint
from yolo.data import prepare_dataset
from yolo.state import cpu_state, state_digest
from yolo.runtime import DEFAULTS


def test_trainer_loads_global_weights_strictly(dataset, names, tmp_path):
    manifest, global_file, _ = prepare_dataset(dataset, names, tmp_path / 'data')
    model = initialize('yolo11n.yaml', names, 42)
    state = cpu_state(model)
    first = next(key for key, value in state.items() if value.is_floating_point())
    state[first].fill_(0.125)
    trainer = FederatedTrainer(model.yaml, state, names, overrides={
        'model': 'yolo11n.yaml', 'data': manifest[0]['data'], 'device': 'cpu',
        'project': str(tmp_path / 'train'), 'name': 'load', 'imgsz': 64,
        'amp': False, 'workers': 0,
    })
    loaded = trainer.get_model()
    assert state_digest(cpu_state(loaded)) == state_digest(state)
    trainer.model = loaded
    trainer._check_start(trainer)
    trainer._capture_last(trainer)
    assert state_digest(trainer.final_state) == state_digest(state)
    with torch.no_grad():
        next(loaded.parameters()).add_(1)
    assert state_digest(trainer.final_state) == state_digest(state)
    trainer.model = loaded
    with pytest.raises(ValueError, match='changed'):
        trainer._check_start(trainer)


def test_saved_global_checkpoint_reopens_for_inference(names, tmp_path):
    from ultralytics import YOLO
    model = initialize('yolo11n.yaml', names, 42)
    path = tmp_path / 'global.pt'
    save_checkpoint(model, path, 'not_used_for_inference.yaml', {**DEFAULTS, 'imgsz': 64}, 1)
    reloaded = YOLO(str(path))
    assert reloaded.names == names
    assert state_digest(cpu_state(reloaded.model)) == state_digest(cpu_state(model))
    import numpy as np
    result = reloaded.predict(np.zeros((64, 64, 3), dtype=np.uint8), imgsz=64, device='cpu', verbose=False)
    assert len(result) == 1
