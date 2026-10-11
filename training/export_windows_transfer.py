"""Build a portable private ZIP from the original Mac project; resolves symlinks."""
import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose another filename')
    files = []
    source = args.source.resolve()
    config = yaml.safe_load((source/'configs/default.yaml').read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        def add_bytes(name, data):
            archive.writestr(name, data)
            files.append({'path': name, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})

        def add_path(name, path):
            # Read actual bytes, never archive Mac symlinks.
            digest = hashlib.sha256()
            with Path(path).open('rb') as stream, archive.open(name, 'w', force_zip64=True) as target:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(block)
                    target.write(block)
            files.append({'path': name, 'sha256': digest.hexdigest(), 'bytes': Path(path).stat().st_size})

        data_root = source/'dataset/edgevision/yolo'
        data_cfg = yaml.safe_load((data_root/'dataset.yaml').read_text())
        assert data_cfg['names'] == {0: 'helmet', 1: 'no_helmet'}
        removed = []
        for folder in ('images', 'labels'):
            for split in ('train', 'val', 'test'):
                for path in sorted((data_root/folder/split).glob('*')):
                    if path.is_file():
                        name = f'dataset/helmet/{folder}/{split}/{path.name}'
                        if folder == 'labels':
                            valid = []
                            for number, line in enumerate(path.read_text().splitlines(), 1):
                                values = list(map(float, line.split()))
                                if len(values) != 5 or values[0] not in (0, 1) or not all(math.isfinite(v) and 0 <= v <= 1 for v in values[1:]):
                                    raise ValueError(f'Invalid source label: {path}:{number}')
                                if min(values[3:]) == 0:
                                    removed.append({'path': name, 'line': number, 'label': line,
                                                    'reason': 'zero-area box cannot represent a physical head'})
                                else:
                                    valid.append(line)
                            add_bytes(name, ('\n'.join(valid)+'\n' if valid else '').encode())
                        else:
                            add_path(name, path)
        add_bytes('dataset/helmet/LABEL-CLEANING.json', json.dumps({'removed': removed,
                  'policy': 'Only mathematically zero-area boxes removed. Valid labels, images and split membership retained. No image content or model results used to edit labels.'}, indent=2).encode())
        add_bytes('dataset/helmet/dataset.yaml', yaml.safe_dump({'train': 'images/train', 'val': 'images/val',
                  'test': 'images/test', 'names': data_cfg['names']}).encode())
        add_path('dataset/helmet/ATTRIBUTION.md', source/'dataset/edgevision/README.md')
        models = sorted({v for section in ('detector', 'classifier') for k,v in config[section].items() if k.endswith('model') and v})
        for model in models:
            add_path(model, source/model)
        add_path('models/SOURCES.md', source/'models/README.md')
        frozen = source/'dataset/head-union-native-scale-20261011/COMBINED-VALIDATION.json'
        original = json.loads(frozen.read_text())
        rows = []
        for index, row in enumerate(original['rows']):
            image = Path(row['image'])
            if not image.is_absolute():
                image = source/image
            assert hashlib.sha256(image.read_bytes()).hexdigest() == row['image_sha256']
            name = f'images/frame-{index:02d}{image.suffix}'
            add_path('evaluation/'+name, image)
            rows.append({**row, 'image': name})
        add_bytes('evaluation/validation.json', json.dumps({'rows': rows,
                  'source_manifest_sha256': hashlib.sha256(frozen.read_bytes()).hexdigest(),
                  'scope': 'Existing fully reviewed VAL only. Not an untouched generalization test.'}, indent=2).encode())
        train_hashes = {r['sha256'] for r in files if r['path'].startswith('dataset/helmet/images/train/')}
        if any(row['image_sha256'] in train_hashes for row in rows):
            raise ValueError('Frozen full-frame validation overlaps transferred detector training images')
        archive.writestr('TRANSFER-MANIFEST.json', json.dumps({'files': files,
             'scope': 'Existing EdgeVision splits, current configured app models and frozen 16 VAL frames; no user screenshots or incomplete broad test queue.',
             'symlinks_materialized': True, 'frozen_val_training_overlap': False}, indent=2))
    print(json.dumps({'output': str(args.output), 'files': len(files), 'bytes': args.output.stat().st_size}))


if __name__ == '__main__':
    main()
