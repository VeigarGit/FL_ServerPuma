import pytest
import torch

from yolo.state import fedavg, signature, state_digest
from yolo.runtime import validate_response


def states(value):
    return {'weight': torch.tensor([value], dtype=torch.float32),
            'bn.running_mean': torch.tensor([value + 1], dtype=torch.float32),
            'bn.num_batches_tracked': torch.tensor(7)}


def test_weighted_fedavg_and_integer_buffers():
    base = states(0)
    result = fedavg(base, [states(2), states(6)], [1, 3])
    assert result['weight'].item() == 5
    assert result['bn.running_mean'].item() == 6
    assert result['bn.num_batches_tracked'].item() == 7
    assert base['weight'].item() == 0


@pytest.mark.parametrize('change', ['missing', 'shape', 'nan', 'dtype'])
def test_invalid_updates_rejected(change):
    update = states(1)
    if change == 'missing':
        del update['weight']
    elif change == 'shape':
        update['weight'] = torch.ones(2)
    elif change == 'nan':
        update['weight'][0] = float('nan')
    else:
        update['weight'] = update['weight'].half()
    with pytest.raises(ValueError):
        fedavg(states(0), [update], [2])


def test_bad_counts_rejected():
    for counts in ([0], [-1], [1.5], [True], []):
        with pytest.raises(ValueError):
            fedavg(states(0), [states(1)], counts)


def test_client_failure_or_wrong_round_cannot_commit():
    base = states(0)
    task = {'state': base, 'round': 2, 'camera': 'camera_01', 'num_images': 2,
            'signature': signature(base, {0: 'person'})}
    with pytest.raises(RuntimeError, match='Client failed'):
        validate_response(task, {'status': 'error', 'error': 'out of memory'})
    response = {**task, 'state': states(1), 'status': 'ok', 'start_digest': state_digest(base)}
    response['round'] = 1
    with pytest.raises(ValueError, match='round'):
        validate_response(task, response)
    response['round'] = 2
    response['start_digest'] = state_digest(states(4))
    with pytest.raises(ValueError, match='global weights'):
        validate_response(task, response)
    assert base['weight'].item() == 0
