# Verification on 2026-10-11

- 169 app and portable-workflow tests passed on the Mac.
- New Python entrypoints passed Ruff and bytecode compilation.
- The complete training preflight ran on CPU with real dense/sparse train/validation fixtures, finite losses and weights, checkpoint save/reload and inference. CPU preflight is explicitly rejected by the CUDA long-training gate.
- Detector and complete-app evaluators returned identical scores when comparing a checkpoint against itself on the frozen 16 validation frames.
- 4,813 transfer files passed SHA-256 verification. One mathematically zero-area source annotation was removed and recorded; other labels and split assignments were retained.
- Release-part assembly tests cover byte order and corrupted-download rejection.

Windows PowerShell execution, CUDA speed, GPU memory use and Windows package installation have not been tested on an actual Windows machine here. Run `windows/setup.ps1` and the CUDA preflight on the RTX PC before longer training. These checks validate portability and workflow mechanics, not unseen-image/video accuracy.
