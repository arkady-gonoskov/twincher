# Twincher

Twincher is a class of ML systems for perception, action planning and other tasks that can be formulated as inverse problems. Instead of approximating intended solutions directly, twinchers learn a representation space that guides a fast, never-stalling iterative approach to the solutions, commonly reaching the needed accuracy within a few iterations. Twinchers learn such representations by exploring a given forward process, such as an image renderer, a mathematical model, a physics simulation or an ML model. They can deal with both well- and ill-posed inverse problems and can be made tolerant to noise in the incoming data.

This package provides the computational primitives of twinchers, with a C++ core that has CPU (OpenMP) and GPU (CUDA) backends and operates PyTorch, NumPy or CuPy arrays, and an example architecture named "hyper-sail" (`hs`).

- Paper: [A. Gonoskov, arXiv:2605.13470 (2026)](https://arxiv.org/abs/2605.13470)
- Website: [www.twincher.ai](https://www.twincher.ai)
- Source code and documentation: [github.com/arkady-gonoskov/twincher](https://github.com/arkady-gonoskov/twincher)
- Contact: [contact@twincher.ai](mailto:contact@twincher.ai)

## Installation

```bash
pip install twincher
```

Twincher requires Python 3.10 or newer and a C++ compiler supporting C++17, since the package is compiled during the installation. The primary platform is Linux (including WSL).

The GPU backend is compiled if the CUDA compiler `nvcc` is found during the installation (for example, if the `bin` directory of the CUDA toolkit is in `PATH`); otherwise twincher is installed for the CPU only. To use a GPU with `twincher.Learner`, PyTorch with CUDA support is needed as well. `python -c "import twincher; twincher.info()"` shows whether the CUDA backend was built and whether a GPU is available.

## Example

Learning to find the position `p` along a spiral from the coordinates `y` of a point on it:

```python
import math
import numpy as np
import twincher

class Spiral:
    """Forward process y(p): position p on a spiral -> point y in the plane."""
    n_p, n_y, name = 1, 2, "spiral"
    def __call__(self, p, y):
        r = 1/(0.5*p[0] + 1.5)
        alpha = 0.25*math.pi + 1.3*math.pi*(p[0] + 1)
        y[0], y[1] = r*math.cos(alpha), r*math.sin(alpha)

stencil = Spiral()

# Learn a representation in which the inverse problem is solved by Gauss-Newton iteration
learner = twincher.Learner(arch="hs", n_s=16, n_l=64)
learner.generate_data(stencil=stencil, n_grid=128)
for _ in range(5001):
    learner.step()
learner.SH.save("spiral.twc")

# Solve the inverse problem y(p) = y_true for 1024 random points
solver = twincher.solver("spiral.twc", n_c=1024)
p_true = np.random.default_rng(0).uniform(-1, 1, size=(1024, 1))
y_true = np.empty((1024, 2))
for p, y in zip(p_true, y_true):
    stencil(p, y)
p_solution = np.empty((1024, 1))
solver.solve(stencil, y_true, n_iter=8, out=p_solution)
print("maximal error of p:", np.abs(p_solution - p_true).max())
```

Diagnostic outputs, such as the learning curve, are written to the directory `twincher_output`. The theory, further examples and the interfaces are described in the paper and in the [README](https://github.com/arkady-gonoskov/twincher#readme) of the project repository.

## License

Twincher is distributed under the GNU Affero General Public License, version 3 (AGPL-3.0). Commercial licenses are available on request: [contact@twincher.ai](mailto:contact@twincher.ai).

## Citation

```bibtex
@misc{gonoskov2026twincher,
  title         = {Twincher: Bijective Representation Learning for Robust Inversion of Continuous Systems},
  author        = {Gonoskov, Arkady},
  year          = {2026},
  eprint        = {2605.13470},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2605.13470},
}
```
