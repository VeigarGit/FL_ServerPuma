import json
import subprocess
import sys

import pytest
import torch

from conftest import ROOT
from yolo.run import parser
from yolo.runtime import run
from yolo.state import state_digest


def test_failed_round_keeps_last_committed_checkpoint(dataset, tmp_path, monkeypatch):
    from yolo import backend, runtime
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.zero_()
    model.yaml = {'nc': 8}
    monkeypatch.setattr(backend, 'initialize', lambda *args: model)
    splits = []

    def evaluate(*args, split='val', **kwargs):
        splits.append(split)
        return {'map50': 0.1, 'map50_95': 0.1, 'precision': 0.1, 'recall': 0.1, 'classes': {}}

    def update(listener, task, *args):
        if task['round'] == 2 and task['camera'] == 'camera_02':
            raise RuntimeError('simulated client failure')
        return {'state': {key: tensor + 1 for key, tensor in task['state'].items()},
                'start_digest': state_digest(task['state']), 'training_seconds': 0.1}, {
                    'sent_bytes': 10, 'received_bytes': 10, 'exchange_seconds': 0.1}

    monkeypatch.setattr(backend, 'evaluate', evaluate)
    monkeypatch.setattr(runtime, 'collect_update', update)
    args = parser().parse_args(['--config', str(ROOT / 'configs/yolo.yaml'),
        '--dataset-root', str(dataset), '--rounds', '2', '--device', 'cpu',
        '--output', str(tmp_path / 'results')])
    with pytest.raises(RuntimeError, match='simulated client failure'):
        run(args)
    output = next((tmp_path / 'results').iterdir())
    assert len(json.loads((output / 'metrics.json').read_text())) == 1
    assert not (output / 'checkpoints/round_002.pt').exists()
    assert not (output / 'test.json').exists()
    assert set(splits) == {'val'}
    for checkpoint in ('last_global.pt', 'best_global.pt', 'checkpoints/round_001.pt'):
        assert torch.load(output / checkpoint)['model'].weight.item() == 1


@pytest.mark.parametrize('entry, flag', [('server', '--model=yolo'), ('client', '--model')])
def test_dispatch_does_not_import_legacy_dependencies(entry, flag):
    command = [sys.executable, str(ROOT / f'src/system/{entry}.py'), flag]
    if flag == '--model':
        command.append('yolo')
    command.append('--help')
    result = subprocess.run(command, text=True, capture_output=True, check=True, cwd=ROOT)
    assert 'yolo' in result.stdout.lower()
    assert not result.stderr
