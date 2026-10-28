# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Loss term `HSSMNoiseLoss` (`hs_sm_noise`, single mode), an alternative to `HSNoiseLoss` for tolerance to noise in `y`. Instead of random directions perpendicular to the manifold `y(p)`, it penalizes, for each case, the perpendicular direction in which `r` changes most per unit change of `y` (the largest singular value of `∂r/∂y` restricted to these directions), so that near the manifold `r` depends only on `p`. It is not used by default; the double-Gaussian tutorial shows how to enable it (`python double_gaussian.py --noise-loss hs_sm_noise`).
- `HSLoss`: optional momentum reset against loss spikes (`momentum_reset`, class `MomentumReset`). A batch with a sudden rise of the number of negative determinants sets the parameters back to the last accepted state, zeroes Adam's first moment and lowers the learning rate temporarily. Adam otherwise turns a spike of the gradient into a displacement of each affected parameter by tens of learning rates. Off by default: it is meant for the late stages of a training and for the continued training of a trained twincher; while the curriculum box grows, isolated negative determinants at its border are normal, and resets there slow the training down.
- `HSLoss`: options `ramp_ext` (cases also drawn beyond the curriculum box, with an amplitude decreasing to zero), `promotion_tol` (a promotion tolerates a small fraction of determinants below `det_clearance`), `p_range_init` (initial size of the box) and `ramp_mode` (`"sum"`: the ramp of version 0.1.0, to reproduce earlier trainings).

### Changed

- `HSLoss` also applies the trimmer to `n_c` extra cases drawn uniformly over the whole domain `[-1, 1]^n_p`, in a second pass (`trimmer_domain_amp`, default 1; 0 restores the previous behaviour). Outside the curriculum box nothing constrained `r` before, and large derivatives there made trainings collapse when the box grew into them. The extra pass increases the time per iteration (by half in the spiral test), and the curriculum reached the whole domain sooner there (after 1653 instead of 2945 iterations).
- `HSLoss`: the amplitude of the determinant loss now changes linearly over the margin of width `ramp_size` and reaches `amp * ramp_boost` at the border of the box. Before, the boost grew with the sum over the components of the distance into the margin, without division by `ramp_size`, so that the default `ramp_boost = 2` raised the amplitude by at most 20 % per component. The default of `ramp_boost` is now 1 (the same amplitude everywhere).
- `HSNoiseLoss` computes the pseudoinverse of `dy/dp` from eigendecompositions of small `n_p x n_p` matrices instead of batched SVDs, which are slow on GPUs (on a laptop GPU, 2 times faster for `n_y = 32, n_p = 2` and about 900 times faster for `n_y = 128, n_p = 4`).
- `HSNoiseLoss` and `HSSMNoiseLoss` warn if `n_y <= n_p`: then `y` has no directions perpendicular to the manifold `y(p)`, and the term would work against the other loss terms.
- The double-Gaussian tutorial also tests the tolerance to noise (`noise_tol.png`) and can be trained with either noise loss term (option `--noise-loss`).

### Fixed

- `HSNoiseLoss`: the random offsets from the manifold `y(p)` set by `y_range` were applied along the tangent directions instead of perpendicular to them.
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
