"""Validate existing camera splits and generate manifests without moving data."""
import hashlib
import math
from pathlib import Path

import yaml
from PIL import Image

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
SPLITS = ('train', 'val', 'test')


def prepare_dataset(root, names, output, cameras=None, allow_missing_labels=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not root.is_dir():
        raise ValueError(f'Dataset root does not exist: {root}')
    available = {p.name: p for p in root.iterdir() if p.is_dir() and (p / 'images').is_dir()}
    selected = sorted(available if cameras is None else cameras)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError('Select at least one camera, without duplicates')
    unknown = set(selected) - available.keys()
    if unknown:
        raise ValueError(f'Unknown cameras: {sorted(unknown)}')
    if sorted(names) != list(range(len(names))) or len(names) != 8:
        raise ValueError('Expected exactly eight class IDs, numbered 0 through 7')
    seen, records, missing, errors = {}, {}, [], []
    for camera in selected:
        folder = available[camera]
        counts, objects = {}, {}
        for split in SPLITS:
            images_dir, labels_dir = folder / 'images' / split, folder / 'labels' / split
            if not images_dir.is_dir() or not labels_dir.is_dir():
                errors.append(f'{camera}/{split}: images and labels directories are required')
                continue
            images = sorted(p for p in images_dir.rglob('*') if p.suffix.lower() in EXTENSIONS and p.is_file())
            counts[split], objects[split] = len(images), 0
            if not images:
                errors.append(f'{camera}/{split}: empty image split')
            expected_labels = set()
            for image in images:
                try:
                    with Image.open(image) as im:
                        im.verify()
                except Exception as exc:
                    errors.append(f'{image}: unreadable image ({exc})')
                digest = hashlib.sha256(image.read_bytes()).hexdigest()
                if digest in seen and seen[digest][0] != split:
                    errors.append(f'Content duplicated across splits: {seen[digest][1]} and {image}')
                else:
                    seen.setdefault(digest, (split, str(image)))
                label = labels_dir / image.relative_to(images_dir).with_suffix('.txt')
                expected_labels.add(label)
                if not label.is_file():
                    missing.append(str(label))
                    if not allow_missing_labels:
                        errors.append(f'{label}: missing label (use empty .txt for negative images)')
                    continue
                for line_number, line in enumerate(label.read_text().splitlines(), 1):
                    if not line.strip():
                        continue
                    try:
                        parts = line.split()
                        if len(parts) != 5:
                            raise ValueError('expected class x_center y_center width height')
                        class_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:])
                        if class_id not in names:
                            raise ValueError('class ID outside 0..7')
                        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y, w, h)) or w <= 0 or h <= 0:
                            raise ValueError('invalid normalized coordinates')
                        if x - w / 2 < -1e-6 or y - h / 2 < -1e-6 or x + w / 2 > 1 + 1e-6 or y + h / 2 > 1 + 1e-6:
                            raise ValueError('bounding box extends outside image')
                        objects[split] += 1
                    except ValueError as exc:
                        errors.append(f'{label}:{line_number}: {exc}')
            orphaned = set(labels_dir.rglob('*.txt')) - expected_labels
            errors.extend(f'{p}: label without corresponding image' for p in sorted(orphaned))
        if objects.get('train', 0) == 0:
            errors.append(f'{camera}: training split needs at least one annotated object')
        records[camera] = {'path': str(folder), 'images': counts, 'objects': objects}
    report = {'root': str(root), 'cameras': records, 'missing_labels': missing, 'errors': errors,
              'temporal_split_verified': False}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'validation_report.yaml').write_text(yaml.safe_dump(report, sort_keys=False))
    if errors:
        raise ValueError(f'Dataset validation failed ({len(errors)} errors). See {output / "validation_report.yaml"}\n' + '\n'.join(errors[:10]))
    manifest = []
    for index, camera in enumerate(selected):
        data = {'path': records[camera]['path'], **{s: f'images/{s}' for s in SPLITS}, 'names': names}
        file = output / f'client_{index:03d}.yaml'
        file.write_text(yaml.safe_dump(data, sort_keys=False))
        manifest.append({'client_id': index, 'camera': camera, 'data': str(file),
                         'num_images': records[camera]['images']['train']})
    global_data = {'path': str(root), 'names': names}
    global_data.update({s: [str(available[c] / 'images' / s) for c in selected] for s in SPLITS})
    global_file = output / 'global.yaml'
    global_file.write_text(yaml.safe_dump(global_data, sort_keys=False))
    (output / 'clients.yaml').write_text(yaml.safe_dump(manifest, sort_keys=False))
    return manifest, str(global_file), report
