"""Strict model contracts and full precision FedAvg."""
import hashlib
import json

import torch


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def signature(state, names):
    schema = [(key, list(value.shape), str(value.dtype)) for key, value in sorted(state.items())]
    return hashlib.sha256(json.dumps([schema, sorted(names.items())]).encode()).hexdigest()


def state_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_state(state, reference):
    if not isinstance(state, dict) or state.keys() != reference.keys():
        raise ValueError('Model state keys do not match global architecture')
    for key, ref in reference.items():
        value = state[key]
        if not isinstance(value, torch.Tensor) or value.shape != ref.shape or value.dtype != ref.dtype:
            raise ValueError(f'Incompatible model tensor: {key}')
        if not torch.isfinite(value).all():
            raise ValueError(f'Non-finite model tensor: {key}')


def fedavg(reference, updates, sample_counts):
    if not updates or len(updates) != len(sample_counts):
        raise ValueError('Updates and sample counts must be non-empty and have equal length')
    if any(type(n) is not int or n <= 0 for n in sample_counts):
        raise ValueError('Sample counts must be positive integers')
    for state in updates:
        validate_state(state, reference)
    total = sum(sample_counts)
    result = {}
    for key, ref in reference.items():
        if ref.is_floating_point():
            value = torch.zeros(ref.shape, dtype=torch.float32)
            for update, count in zip(updates, sample_counts):
                value.add_(update[key].detach().cpu().float(), alpha=count / total)
            result[key] = value.to(ref.dtype)
        else:
            result[key] = ref.detach().cpu().clone()
    validate_state(result, reference)
    return result
