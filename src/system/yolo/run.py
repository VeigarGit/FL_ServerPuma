"""CLI shared by the launcher and early server/client dispatch."""
import argparse
import os
from pathlib import Path
import sys

SYSTEM_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = SYSTEM_DIR.parents[1]
for directory in (SYSTEM_DIR, SYSTEM_DIR.parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
os.environ.setdefault('YOLO_CONFIG_DIR', str(REPO_DIR / '.yolo-settings'))
os.environ.setdefault('MPLCONFIGDIR', str(REPO_DIR / '.yolo-settings' / 'matplotlib'))
os.environ.setdefault('WANDB_DISABLED', 'true')


def parser():
    cli = argparse.ArgumentParser(description='YOLO11n FedAvg: one camera per client, sequential GPU training')
    cli.add_argument('--model', choices=['yolo'], default='yolo')
    cli.add_argument('--config', default='configs/yolo.yaml')
    cli.add_argument('--dataset-root', required=True)
    cli.add_argument('--cameras', nargs='+')
    cli.add_argument('--rounds', type=int)
    cli.add_argument('--local-epochs', type=int)
    cli.add_argument('--batch', type=int)
    cli.add_argument('--imgsz', type=int)
    cli.add_argument('--device')
    cli.add_argument('--weights')
    cli.add_argument('--output')
    cli.add_argument('--timeout', type=float)
    cli.add_argument('--allow-missing-labels', action='store_true', default=None)
    cli.add_argument('--validate-only', action='store_true')
    return cli


def main():
    args = parser().parse_args()
    from yolo.runtime import run
    try:
        run(args)
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(f'YOLO execution failed: {exc}') from exc


def client_main():
    cli = argparse.ArgumentParser(description='Internal one-task YOLO TCP client')
    cli.add_argument('--model', choices=['yolo'], default='yolo')
    cli.add_argument('--host', default='127.0.0.1', choices=['127.0.0.1'])
    cli.add_argument('--port', required=True, type=int)
    cli.add_argument('--timeout', type=float, default=3600)
    args = cli.parse_args()
    from yolo.runtime import client
    client(args)


if __name__ == '__main__':
    main()
