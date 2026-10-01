# Tutorials

The scripts in this directory reproduce the examples of the hyper-sail architecture (`hs`) discussed in the [README](../README.md). They use a GPU if one is available (PyTorch with CUDA support and twincher built with CUDA) and the CPU otherwise.

Run them from this directory:

```bash
cd tutorial
python spiral.py
python double_gaussian.py
```

Each script writes its results to a directory next to it: `output_spiral_1.3` and `output_double_gaussian`, respectively.

## Spiral: `spiral.py`

A well-posed problem with one parameter: finding the position `p` along a spiral from the coordinates `y` of a point on it (`n_p = 1`, `n_y = 2`). The script trains an `hs` twincher for 5001 iterations with data generated from the stencil on a grid, and then evaluates it. Outputs:

- `0.png`, `1.png`, ...: the learned representation r<sub>0</sub>(y<sub>0</sub>, y<sub>1</sub>) every 200 iterations (Fig. 3 of the README);
- `lc.png`: learning curve;
- `final.twc`: the trained twincher;
- `spiral_parity_plot-light.svg`: solutions found with the trained twincher and by Gauss-Newton descent in `y` space, against the true values (Fig. 4);
- `spiral_1.3_noise_tol.png`: tolerance to noise in `y` (Fig. 5).

## Double Gaussian: `double_gaussian.py`

A problem with two parameters: finding the positions `p` of two Gaussian peaks from values of their sum on a grid of 32 points (`n_p = 2`, `n_y = 32`). The twincher is trained for 13001 iterations, computing data from the stencil during the training. Afterwards, the inverse problem is solved for 5120 random cases and the errors of `p` are printed. Outputs:

- `example.png`: an example of the distribution (Fig. 8 of the README);
- `000.png`, `001.png`, ...: the grid spanning `p` in the learned representation `r` every 200 iterations (Fig. 9);
- `lc.png`: learning curve;
- `final.twc`: the trained twincher.

If `final.twc` already exists, the training is skipped and only the solution is tested. Delete the file to train again.

## Runtimes

Measured on a laptop with an Intel Core i5-13450HX (16 threads) and an NVIDIA GeForce RTX 5050 Laptop GPU:

| Script | GPU | CPU only |
|---|---|---|
| `spiral.py` | 1.5 minutes | 9 minutes |
| `double_gaussian.py` | 8 minutes | about 2 hours (estimated) |

## Helper modules

- `output_tools_p1_y2.py`: plots for problems with `n_p = 1` and `n_y = 2` (state of the representation, parity plot). The variable `COLOR_SCHEME` selects a light or dark style.
- `output_tools_p2.py`: plot of an example of the double-Gaussian distribution.
