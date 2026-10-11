"""Fail early when CUDA is unavailable; save machine details for troubleshooting."""
import json
import platform
from pathlib import Path

import torch


def main():
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable. Update the NVIDIA Windows driver and run windows/setup.ps1. Do not start a CPU training run by accident.')
    device = torch.cuda.get_device_properties(0)
    sample = torch.randn(16, 16, device='cuda', requires_grad=True)
    sample.square().mean().backward()
    torch.cuda.synchronize()
    assert torch.isfinite(sample.grad).all()
    report = {'gpu': device.name, 'vram_gib': round(device.total_memory/2**30, 2),
              'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
              'python': platform.python_version(), 'platform': platform.platform(),
              'gradient_smoke_passed': True}
    output = Path('data/outputs')
    output.mkdir(parents=True, exist_ok=True)
    (output/'cuda-check.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
