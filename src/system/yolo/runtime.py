"""Single-host TCP orchestration, one GPU client at a time, atomic round commits."""
import csv
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

import torch
import yaml

from utils.network_utils import recv_data, send_data
from .data import prepare_dataset
from .state import cpu_state, fedavg, signature, state_digest, validate_state

REPO = Path(__file__).resolve().parents[3]
DEFAULTS = {
    'weights': 'yolo11n.pt', 'rounds': 20, 'local_epochs': 1, 'imgsz': 640,
    'batch': 4, 'optimizer': 'AdamW', 'lr0': 0.001, 'seed': 42, 'amp': False,
    'device': '0', 'workers': 0, 'output': 'src/results/yolo',
    'allow_missing_labels': False, 'timeout': 3600,
}


def load_config(args):
    raw = yaml.safe_load(Path(args.config).read_text())
    if not isinstance(raw, dict):
        raise ValueError('Config must be a YAML mapping')
    unknown = raw.keys() - (DEFAULTS.keys() | {'names'})
    if unknown:
        raise ValueError(f'Unknown config options: {sorted(unknown)}')
    config = {**DEFAULTS, **raw}
    for key in ('rounds', 'local_epochs', 'batch', 'imgsz', 'device', 'weights', 'output', 'timeout', 'allow_missing_labels'):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    for key in ('rounds', 'local_epochs', 'batch', 'imgsz'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if config['imgsz'] % 32:
        raise ValueError('imgsz must be a multiple of 32')
    if type(config['seed']) is not int or type(config['workers']) is not int or config['workers'] < 0:
        raise ValueError('seed and workers must be integers; workers cannot be negative')
    for key in ('lr0', 'timeout'):
        if not isinstance(config[key], (int, float)) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ('amp', 'allow_missing_labels'):
        if type(config[key]) is not bool:
            raise ValueError(f'{key} must be a YAML boolean')
    if config['optimizer'] != 'AdamW':
        raise ValueError('This version uses optimizer AdamW')
    if not isinstance(config.get('names'), dict):
        raise ValueError('Config needs the eight class names')
    config['names'] = {int(key): value for key, value in config['names'].items()}
    if not all(isinstance(v, str) and v.strip() for v in config['names'].values()):
        raise ValueError('Class names must be non-empty strings')
    config['device'] = str(config['device'])
    if config['device'] != 'cpu' and not config['device'].isdigit():
        raise ValueError('device must be cpu or a single CUDA GPU index such as 0')
    return config


def release_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validate_response(task, response):
    if not isinstance(response, dict) or response.get('status') != 'ok':
        raise RuntimeError(f'Client failed: {response}')
    for key in ('round', 'camera', 'signature', 'num_images'):
        if response.get(key) != task[key]:
            raise ValueError(f'Client response has inconsistent {key}')
    if response.get('start_digest') != state_digest(task['state']):
        raise ValueError('Client did not start with the global weights of this round')
    validate_state(response.get('state'), task['state'])
    return response


def collect_update(listener, task, log_path, timeout):
    command = [sys.executable, str(REPO / 'src/system/client.py'), '--model', 'yolo',
               '--port', str(listener.getsockname()[1]), '--timeout', str(timeout)]
    process = None
    with log_path.open('w') as log:
        try:
            process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            listener.settimeout(min(60, timeout))
            conn, _ = listener.accept()
            with conn:
                conn.settimeout(timeout)
                started = time.perf_counter()
                sent = send_data(conn, task)
                response, _, received = recv_data(conn, return_bytes=True)
                elapsed = time.perf_counter() - started
            code = process.wait(timeout=timeout)
            if code:
                raise RuntimeError(f'Client exited with status {code}; see {log_path}')
            response = validate_response(task, response)
            return response, {'sent_bytes': sent, 'received_bytes': received,
                              'exchange_seconds': elapsed, 'log': str(log_path)}
        except Exception as exc:
            raise RuntimeError(f'Camera {task["camera"]}, round {task["round"]} failed; see {log_path}: {exc}') from exc
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def client(args):
    with socket.create_connection((args.host, args.port), timeout=args.timeout) as conn:
        conn.settimeout(args.timeout)
        task, _ = recv_data(conn)
        if task is None:
            raise RuntimeError('Server disconnected before sending training task')
        try:
            if signature(task['state'], task['names']) != task['signature']:
                raise ValueError('Global model signature is invalid')
            from .backend import train_local
            update = train_local(task)
            response = {**update, 'status': 'ok', **{key: task[key] for key in ('round', 'camera', 'signature', 'num_images')}}
        except Exception as exc:
            send_data(conn, {'status': 'error', 'error': str(exc)})
            raise
        send_data(conn, response)


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def write_metrics(output, rounds):
    write_json(output / 'metrics.json', rounds)
    with (output / 'metrics.csv').open('w', newline='') as stream:
        fields = ['round', 'map50', 'map50_95', 'precision', 'recall', 'round_seconds', 'sent_bytes', 'received_bytes']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in rounds:
            writer.writerow({'round': record['round'], **{key: record['global'][key] for key in fields[1:5]},
                             **{key: record[key] for key in fields[5:]}})


def run(args):
    config = load_config(args)
    output = Path(config['output']).resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    config.update(dataset_root=str(Path(args.dataset_root).resolve()), cameras=args.cameras, output=str(output))
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    print(f'Results: {output}', flush=True)
    clients, global_data, report = prepare_dataset(
        args.dataset_root, config['names'], output / 'data', args.cameras, config['allow_missing_labels'])
    if args.validate_only:
        print(f'Dataset valid: {len(clients)} cameras. Missing labels: {len(report["missing_labels"])}', flush=True)
        return output
    if config['device'] != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Use --device cpu for a small smoke test, or install the NVIDIA CUDA runtime.')
    from .backend import evaluate, initialize, save_checkpoint
    model = initialize(config['weights'], config['names'], config['seed'])
    reference = cpu_state(model)
    model_signature = signature(reference, config['names'])
    rounds, best_score, best_round = [], -1.0, None
    baseline = evaluate(model, global_data, config, output / 'evaluation' / 'baseline')
    write_json(output / 'baseline.json', baseline)
    release_gpu()
    (output / 'logs').mkdir()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        for round_number in range(1, config['rounds'] + 1):
            started = time.perf_counter()
            updates, samples, local_records = [], [], []
            # reference stays immutable until every camera succeeds; all see the same start.
            for info in clients:
                print(f'Round {round_number}/{config["rounds"]}: camera {info["camera"]}', flush=True)
                task = {**info, 'round': round_number, 'signature': model_signature,
                        'state': reference, 'architecture': model.yaml, 'names': config['names'],
                        'config': config, 'local_output': str(output / 'clients' / str(info['client_id']))}
                update, communication = collect_update(
                    listener, task, output / 'logs' / f'round_{round_number:03d}_client_{info["client_id"]:03d}.log', config['timeout'])
                updates.append(update['state'])
                samples.append(info['num_images'])
                local_records.append({**info, **communication, 'training_seconds': update['training_seconds'],
                                      'start_digest': update['start_digest']})
            candidate = fedavg(reference, updates, samples)
            model.load_state_dict(candidate, strict=True)
            global_metrics = evaluate(model, global_data, config, output / 'evaluation' / f'round_{round_number:03d}_global')
            release_gpu()
            for record in local_records:
                record['validation'] = evaluate(model, record['data'], config,
                    output / 'evaluation' / f'round_{round_number:03d}_camera_{record["client_id"]:03d}')
                release_gpu()
            record = {'round': round_number, 'global_start_digest': state_digest(reference),
                      'global_end_digest': state_digest(candidate), 'global': global_metrics, 'clients': local_records,
                      'round_seconds': time.perf_counter() - started,
                      'sent_bytes': sum(item['sent_bytes'] for item in local_records),
                      'received_bytes': sum(item['received_bytes'] for item in local_records)}
            save_checkpoint(model, output / 'checkpoints' / f'round_{round_number:03d}.pt', global_data, config, round_number)
            save_checkpoint(model, output / 'last_global.pt', global_data, config, round_number)
            if global_metrics['map50_95'] > best_score:
                best_score, best_round = global_metrics['map50_95'], round_number
                save_checkpoint(model, output / 'best_global.pt', global_data, config, round_number)
            rounds.append(record)
            write_metrics(output, rounds)
            reference = candidate
            print(f'Round {round_number}: mAP50-95={global_metrics["map50_95"]:.6f}', flush=True)
    from ultralytics import YOLO
    best_model = YOLO(str(output / 'best_global.pt')).model.cpu().float()
    test = {'selected_round': best_round, 'global': evaluate(best_model, global_data, config, output / 'evaluation' / 'final_test_global', split='test'), 'cameras': {}}
    release_gpu()
    for info in clients:
        test['cameras'][info['camera']] = evaluate(best_model, info['data'], config,
            output / 'evaluation' / f'final_test_camera_{info["client_id"]:03d}', split='test')
        release_gpu()
    write_json(output / 'test.json', test)
    print(f'Completed. Best model: {output / "best_global.pt"}', flush=True)
    return output
