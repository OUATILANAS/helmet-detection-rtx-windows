import hashlib
import json
import subprocess
import sys
from pathlib import Path


def run_join(tmp_path, corrupt=False):
    parts = []
    for name, payload in [('one.part', b'first'), ('two.part', b'second')]:
        (tmp_path/name).write_bytes(payload)
        parts.append({'name': name, 'sha256': hashlib.sha256(payload).hexdigest()})
    (tmp_path/'docs').mkdir()
    (tmp_path/'docs/TRANSFER-PARTS.json').write_text(json.dumps({
        'parts': parts, 'zip_sha256': hashlib.sha256(b'firstsecond').hexdigest()}))
    if corrupt:
        (tmp_path/'two.part').write_bytes(b'damaged')
    script = Path(__file__).resolve().parents[1]/'training/join_transfer.py'
    return subprocess.run([sys.executable, str(script), '--parts', str(tmp_path)],
                          cwd=tmp_path, capture_output=True, text=True)


def test_join_preserves_order_and_bytes(tmp_path):
    result = run_join(tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path/'helmet-windows-transfer.zip').read_bytes() == b'firstsecond'


def test_join_rejects_corrupt_download(tmp_path):
    result = run_join(tmp_path, corrupt=True)
    assert result.returncode != 0
    assert 'checksum mismatch' in result.stderr
