import shutil
from pathlib import Path

import pytest
import yaml

from yolo.data import prepare_dataset


def test_camera_splits_and_global_manifests(dataset, names, tmp_path):
    manifest, global_file, report = prepare_dataset(dataset, names, tmp_path / 'data')
    assert [record['camera'] for record in manifest] == ['camera_01', 'camera_02']
    assert [record['num_images'] for record in manifest] == [2, 2]
    data = yaml.safe_load(open(global_file))
    assert data['names'] == names
    assert data['val'] == [str(dataset / c / 'images/val') for c in ('camera_01', 'camera_02')]
    assert all(Path(value).name == 'val' for value in data['val'])
    assert report['temporal_split_verified'] is False
    local = yaml.safe_load(open(manifest[0]['data']))
    assert local['path'] == str(dataset / 'camera_01')


def test_missing_labels_need_explicit_opt_in(dataset, names, tmp_path):
    (dataset / 'camera_01/labels/train/0.txt').unlink()
    with pytest.raises(ValueError, match='missing label'):
        prepare_dataset(dataset, names, tmp_path / 'strict')
    _, _, report = prepare_dataset(dataset, names, tmp_path / 'allowed', allow_missing_labels=True)
    assert len(report['missing_labels']) == 1


def test_duplicate_content_across_cameras_and_splits(dataset, names, tmp_path):
    shutil.copyfile(dataset / 'camera_01/images/train/0.png', dataset / 'camera_02/images/test/0.png')
    with pytest.raises(ValueError, match='duplicated across splits'):
        prepare_dataset(dataset, names, tmp_path / 'data')


@pytest.mark.parametrize('annotation', ['8 0.5 0.5 0.5 0.5', '0 nan 0.5 0.5 0.5',
                                         '0 0.9 0.5 0.5 0.5', '0 0.5 0.5 0 0.5'])
def test_invalid_annotations(dataset, names, tmp_path, annotation):
    (dataset / 'camera_01/labels/train/0.txt').write_text(annotation)
    with pytest.raises(ValueError, match='Dataset validation failed'):
        prepare_dataset(dataset, names, tmp_path / 'data')


def test_camera_selection_and_empty_split(dataset, names, tmp_path):
    manifest, _, _ = prepare_dataset(dataset, names, tmp_path / 'selected', cameras=['camera_02'])
    assert len(manifest) == 1 and manifest[0]['camera'] == 'camera_02'
    for image in (dataset / 'camera_01/images/val').glob('*.png'):
        image.unlink()
    with pytest.raises(ValueError, match='empty image split'):
        prepare_dataset(dataset, names, tmp_path / 'empty')
