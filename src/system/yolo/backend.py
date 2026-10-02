"""Ultralytics 8.3.0 adapter: strict global initialization, raw last-epoch update."""
import copy
from datetime import datetime, timezone
from pathlib import Path
import random
import time

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel

from .state import cpu_state, state_digest, validate_state


def initialize(weights, names, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pretrained = YOLO(weights, task='detect')
    model = DetectionModel(copy.deepcopy(pretrained.model.yaml), nc=len(names), verbose=False)
    model.load(pretrained.model)
    model.names = dict(names)
    model.args = {'task': 'detect', 'model': str(weights), 'imgsz': 640}
    return model.cpu().float()


class FederatedTrainer(DetectionTrainer):
    def __init__(self, architecture, global_state, names, *args, **kwargs):
        self.architecture = copy.deepcopy(architecture)
        self.global_state = global_state
        self.class_names = names
        self.start_digest = None
        self.final_state = None
        super().__init__(*args, **kwargs)
        self.add_callback('on_train_start', self._check_start)
        self.add_callback('on_train_epoch_end', self._capture_last)

    def get_model(self, cfg=None, weights=None, verbose=True):
        if self.data['names'] != self.class_names:
            raise ValueError('Dataset class names differ from global model')
        model = DetectionModel(copy.deepcopy(self.architecture), nc=len(self.class_names), verbose=False)
        validate_state(self.global_state, cpu_state(model))
        model.load_state_dict(self.global_state, strict=True)
        return model

    def _check_start(self, trainer):
        self.start_digest = state_digest(cpu_state(self.model))
        if self.start_digest != state_digest(self.global_state):
            raise ValueError('Trainer changed the received global weights before training')

    def _capture_last(self, trainer):
        # Runs before final_eval strips/reloads checkpoints; aggregate raw weights, not EMA/best.
        self.final_state = cpu_state(self.model)


def train_local(task):
    config = task['config']
    started = time.perf_counter()
    trainer = FederatedTrainer(
        task['architecture'], task['state'], task['names'],
        overrides={
            'model': 'yolo11n.yaml', 'data': task['data'], 'task': 'detect',
            'epochs': config['local_epochs'], 'imgsz': config['imgsz'], 'batch': config['batch'],
            'device': config['device'], 'workers': config['workers'],
            'optimizer': config['optimizer'], 'lr0': config['lr0'],
            'seed': config['seed'], 'amp': config['amp'], 'patience': 0,
            'warmup_epochs': 0.0, 'nbs': config['batch'], 'close_mosaic': 0,
            'resume': False, 'pretrained': False, 'plots': False,
            'project': task['local_output'], 'name': f'round_{task["round"]:03d}',
            'exist_ok': False, 'save': False, 'val': True,
        },
    )
    trainer.train()
    if trainer.final_state is None:
        raise RuntimeError('Trainer did not complete a local epoch')
    validate_state(trainer.final_state, task['state'])
    return {'state': trainer.final_state, 'start_digest': trainer.start_digest,
            'training_seconds': time.perf_counter() - started}


def wrapped_model(model):
    wrapper = YOLO('yolo11n.yaml', task='detect')
    wrapper.model = copy.deepcopy(model).cpu().float()
    return wrapper


def evaluate(model, data, config, output, split='val'):
    result = wrapped_model(model).val(
        data=data, split=split, device=config['device'], imgsz=config['imgsz'],
        batch=config['batch'], workers=config['workers'], plots=False,
        project=str(output.parent), name=output.name, exist_ok=False, half=False,
    )
    metrics = {'map50': float(result.box.map50), 'map50_95': float(result.box.map),
               'precision': float(result.box.mp), 'recall': float(result.box.mr)}
    metrics['classes'] = {
        str(int(class_id)): dict(zip(('precision', 'recall', 'map50', 'map50_95'),
                                     map(float, result.box.class_result(index))))
        for index, class_id in enumerate(result.box.ap_class_index)
    }
    if not all(np.isfinite(metrics[key]) for key in ('map50', 'map50_95', 'precision', 'recall')):
        raise ValueError('Non-finite validation metrics')
    return metrics


def save_checkpoint(model, path, data, config, round_number):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = copy.deepcopy(model).cpu().float()
    snapshot.args = {'task': 'detect', 'data': data, 'imgsz': config['imgsz'], 'model': 'yolo11n.yaml'}
    checkpoint = {'model': snapshot, 'ema': None, 'optimizer': None,
                  'train_args': snapshot.args, 'epoch': round_number,
                  'version': ultralytics.__version__, 'date': datetime.now(timezone.utc).isoformat()}
    temporary = path.with_suffix('.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)
