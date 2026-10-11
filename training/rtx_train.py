"""Bounded preflight, resumable CUDA YOLO training, and explicit test evaluation."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import yaml

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def audit(data_path):
    """Audit train/val only. Never decode test images for training or preflight."""
    cfg = yaml.safe_load(data_path.read_text())
    names = cfg['names']
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    if names != ['helmet', 'no_helmet']:
        raise ValueError('This trainer requires class 0=helmet, 1=no_helmet; do not use physical-head-only data.')
    root = Path(cfg.get('path', '.'))
    if not root.is_absolute():
        root = (data_path.parent/root).resolve()
    rows, seen = {}, {}
    signature = hashlib.sha256()
    for split in ('train', 'val'):
        folder = root/cfg[split]
        rows[split] = []
        if not folder.is_dir():
            raise ValueError(f'Expected image folder: {folder}')
        for image in sorted(folder.rglob('*')):
            if not image.is_file() or image.suffix.lower() not in EXTENSIONS:
                continue
            relative = image.relative_to(folder)
            label = root/'labels'/split/relative.with_suffix('.txt')
            if not label.is_file():
                raise ValueError(f'Missing label file: {label}; use an empty file for a reviewed negative frame')
            for line in label.read_text().splitlines():
                fields = list(map(float, line.split()))
                if len(fields) != 5 or fields[0] not in (0, 1) or not all(0 <= n <= 1 for n in fields[1:]) or min(fields[3:]) <= 0:
                    raise ValueError(f'Invalid YOLO label: {label}')
            image_sha, label_sha = sha(image), sha(label)
            if image_sha in seen and seen[image_sha] != split:
                raise ValueError('Exact train/validation image overlap detected')
            seen[image_sha] = split
            signature.update(f'{split}/{relative.as_posix()}/{image_sha}/{label_sha}\n'.encode())
            rows[split].append((image, label))
        if not rows[split]:
            raise ValueError(f'Empty {split} split')
    return rows, signature.hexdigest()


def signature(args, data_sha):
    return {'data_sha256': data_sha, 'weights_sha256': sha(args.weights),
            'imgsz': args.imgsz, 'batch': args.batch, 'device': args.device,
            'ultralytics_version': __import__('ultralytics').__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['preflight', 'train', 'resume', 'test'])
    parser.add_argument('--data', type=Path, default=Path('dataset/helmet/dataset.yaml'))
    parser.add_argument('--weights', type=Path, default=Path('models/helmet_localizer_video_20261010.pt'))
    parser.add_argument('--imgsz', type=int, default=768)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--name', default='helmet-rtx-768')
    parser.add_argument('--preflight', type=Path, default=Path('data/outputs/rtx-preflight.json'))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', choices=['0', 'cpu'], default='0', help='CPU is for development smoke tests only')
    args = parser.parse_args()
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(2)
    if args.device == '0' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; run training/check_cuda.py before training')
    if min(args.imgsz, args.batch, args.epochs) <= 0:
        parser.error('imgsz, batch and epochs must be positive')
    if args.mode in ('resume', 'test'):
        if not args.checkpoint or not args.checkpoint.is_file():
            parser.error('Provide an existing --checkpoint')
        if args.mode == 'resume':
            if args.device != '0':
                parser.error('Production resume requires CUDA')
            YOLO(args.checkpoint).train(resume=True, device=0, workers=0, cache=False)
        else:
            YOLO(args.checkpoint).val(data=str(args.data.resolve()), split='test',
                  imgsz=args.imgsz, batch=args.batch, device=args.device, workers=0)
        return
    rows, data_sha = audit(args.data)
    settings = signature(args, data_sha)
    args.preflight.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == 'preflight':
        # Real images, full optimizer/backward/save/reload. The model is discarded.
        root = Path('data/outputs/rtx-smoke-data')
        if root.exists():
            shutil.rmtree(root)
        fixture_counts = {}
        for split, pairs in rows.items():
            # Include the densest scene and a sparse/empty scene, not just easy first files.
            def count(pair):
                return len(pair[1].read_text().splitlines())
            ordered = [max(pairs, key=count), min(pairs, key=count), *pairs]
            chosen = list(dict.fromkeys(ordered))[:max(4, args.batch)]
            fixture_counts[split] = [count(pair) for pair in chosen]
            for index, (image, label) in enumerate(chosen):
                for folder in ('images', 'labels'):
                    (root/folder/split).mkdir(parents=True, exist_ok=True)
                shutil.copy2(image, root/'images'/split/f'{index}{image.suffix}')
                shutil.copy2(label, root/'labels'/split/f'{index}.txt')
        data = root/'dataset.yaml'
        data.write_text(yaml.safe_dump({'path': str(root.resolve()), 'train': 'images/train',
                        'val': 'images/val', 'names': {0: 'helmet', 1: 'no_helmet'}}))
        started = time.monotonic()
        if args.device == '0':
            torch.cuda.reset_peak_memory_stats()
        model = YOLO(args.weights)
        model.train(data=str(data.resolve()), epochs=1, imgsz=args.imgsz, batch=args.batch,
                    device=args.device, workers=0, cache=False, amp=args.device == '0',
                    optimizer='AdamW', lr0=.0001, nbs=args.batch, warmup_epochs=0,
                    mosaic=0, plots=False, project=str(Path('runs/preflight').resolve()), name=args.name,
                    exist_ok=True, save=True)
        losses = model.trainer.tloss
        values = losses.values() if isinstance(losses, dict) else [losses]
        assert all(torch.isfinite(torch.as_tensor(value)).all() for value in values)
        checkpoint = Path(model.trainer.save_dir)/'weights/last.pt'
        reloaded = YOLO(checkpoint)
        result = reloaded.predict(str(rows['val'][0][0]), imgsz=args.imgsz,
                                  device=args.device, verbose=False)[0]
        assert torch.isfinite(result.boxes.data).all()
        assert all(torch.isfinite(p).all() for p in reloaded.model.parameters())
        report = {'passed': True, 'settings': settings,
                  'cuda_validated': args.device == '0',
                  'gpu': torch.cuda.get_device_name(0) if args.device == '0' else 'CPU development check',
                  'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30 if args.device == '0' else None,
                  'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30 if args.device == '0' else None,
                  'elapsed_seconds': time.monotonic()-started,
                  'train_images': len(rows['train']), 'val_images': len(rows['val']),
                  'fixture_head_counts': fixture_counts,
                  'test_opened': False, 'model_save_reload_verified': True}
        args.preflight.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
        return
    if args.device != '0':
        parser.error('Long training requires CUDA; CPU is only allowed for preflight verification')
    checked = json.loads(args.preflight.read_text())
    if not checked['passed'] or not checked['cuda_validated'] or checked['settings'] != settings:
        parser.error('Run a passing CUDA preflight with the same data, weights, image size and batch first')
    YOLO(args.weights).train(data=str(args.data.resolve()), epochs=args.epochs,
        imgsz=args.imgsz, batch=args.batch, device=0, workers=0, cache=False,
        amp=True, optimizer='AdamW', lr0=.0001, cos_lr=True, patience=8,
        close_mosaic=5, seed=17, nbs=16, save=True, save_period=1,
        project=str(Path('runs').resolve()), name=args.name)


if __name__ == '__main__':
    main()
