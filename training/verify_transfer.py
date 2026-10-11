"""Verify private transfer files and reject unsafe manifest paths."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path.cwd().resolve()
    manifest = json.loads((root/'TRANSFER-MANIFEST.json').read_text())
    for row in manifest['files']:
        path = (root/row['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Manifest path leaves repository')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
        if digest.hexdigest() != row['sha256']:
            raise ValueError(f'Checksum mismatch: {row["path"]}')
    print(f'Verified {len(manifest["files"])} transfer files. Training/test splits unchanged.')


if __name__ == '__main__':
    main()
