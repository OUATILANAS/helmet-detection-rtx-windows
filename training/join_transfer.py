"""Join and verify the release's split ZIP without loading it into RAM."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts', type=Path, required=True, help='Folder containing every release .part file')
    parser.add_argument('--output', type=Path, default=Path('helmet-windows-transfer.zip'))
    args = parser.parse_args()
    manifest = json.loads(Path('docs/TRANSFER-PARTS.json').read_text())
    paths = [args.parts/p['name'] for p in manifest['parts']]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        parser.error('Download all parts first. Missing: '+', '.join(missing))
    if args.output.exists():
        parser.error('Output already exists; verify it or choose another --output filename')
    whole = hashlib.sha256()
    with args.output.open('xb') as target:
        for path, expected in zip(paths, manifest['parts'], strict=True):
            checksum = hashlib.sha256()
            with path.open('rb') as source:
                for block in iter(lambda: source.read(1024*1024), b''):
                    checksum.update(block)
                    whole.update(block)
                    target.write(block)
            if checksum.hexdigest() != expected['sha256']:
                raise ValueError(f'Part checksum mismatch: {path}; remove the partial output and redownload this part')
    if whole.hexdigest() != manifest['zip_sha256']:
        raise ValueError('Combined ZIP checksum mismatch')
    print(f'Joined and verified {args.output}. Extract it into the repository root.')


if __name__ == '__main__':
    main()
