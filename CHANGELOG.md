# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `HSNoiseLoss` computes the pseudoinverse of `dy/dp` from eigendecompositions of small `n_p x n_p` matrices instead of batched SVDs, which are slow on GPUs (on a laptop GPU, 2 times faster for `n_y = 32, n_p = 2` and about 900 times faster for `n_y = 128, n_p = 4`).

### Fixed

- `HSSolver`: for `n_p = 1`, the in-place inversion of 1x1 matrices returned 1 instead of the inverse, so that the solution converged linearly instead of quadratically.

## [0.1.0] - 2026-10-02

First public release.

### Added

- Computational core in C++ with a CPU backend (OpenMP) and an optional GPU backend (CUDA), accessible through `twincher.Shuttle`. It supports PyTorch tensors, NumPy arrays and CuPy arrays.
- The hyper-sail architecture (`hs`): loss terms `HSLoss` and `HSNoiseLoss`, solver `HSSolver` and monitor `HSMonitorP2`.
- Training and evaluation: `Learner`, `solver()`, `Verifier`, `noise_check()`, `monitor()` and `TimeMonitor`.
- A registry for adding architectures (`register()`).
- File format for trained twinchers (`.twc`) with a format version.
- Tutorials: spiral and double Gaussian.

[Unreleased]: https://github.com/arkady-gonoskov/twincher/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/arkady-gonoskov/twincher/releases/tag/v0.1.0
