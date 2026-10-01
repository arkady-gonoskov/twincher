# Contributing to Twincher

Thank you for considering a contribution to Twincher! Bug reports, suggestions and pull requests are welcome.

## License and Contributor License Agreement

Twincher is distributed under the [GNU Affero General Public License, version 3](LICENSE) (AGPL-3.0). It is also available under separate commercial licenses. So that contributions can be distributed on both of these terms, every contributor needs to accept the [Contributor License Agreement](CLA.md) (CLA) before their first pull request can be merged.

The CLA is a license, not a transfer of ownership: you keep the copyright in your contributions. When you make a contribution to the Project, the Project Owner also commits to keep it available under the open source license of the Project (see Section 4 of the CLA).

You can accept the CLA in either of two ways:

- **Individuals:** confirm it in a comment on your pull request, when asked to do so.
- **Companies and other organizations:** send a signed copy to contact@twincher.ai.

If you contribute as part of your job, please make sure your employer agrees. Your employer may need to accept the CLA as an organization.

## Development setup

Requirements:

- Linux (or WSL);
- Python >= 3.10;
- a C++17 compiler with OpenMP support;
- for the GPU backend, the CUDA toolkit (`nvcc`).

CMake and Ninja are installed into the virtual environment below.

Create a virtual environment and install the build tools into it:

```bash
python -m venv .venv
source .venv/bin/activate
pip install scikit-build-core pybind11 cmake ninja
```

Then install twincher in editable mode:

```bash
pip install --no-build-isolation -Ceditable.rebuild=true -Cbuild-dir=build/editable -e ".[dev]"
```

In this mode, the Python sources are used directly from `python/twincher`. After you change the C++/CUDA sources, the extension module is rebuilt automatically at the next `import twincher`. The build files are kept in `build/editable`, so rebuilds are incremental.

### Running the tests

```bash
python -m pytest                          # all tests (about one minute)
python -m pytest -m "not e2e"             # unit tests only (a few seconds)
python -m pytest -rP tests/test_core.py   # also show the values printed by the tests
```

The tests use the installed package, so they can be run from any environment in which twincher is installed.

Tests that need a GPU are marked with `gpu`. They are skipped automatically if twincher cannot use a GPU: either there is none, or twincher was built without CUDA. The GPU tests with CuPy arrays are additionally skipped if CuPy is not installed (`pip install cupy-cuda13x`, matching your version of CUDA).

| Tests | Content |
|---|---|
| `test_core.py` | The computational core: CPU against finite differences, GPU against CPU |
| `test_shuttle.py`, `test_errors.py`, `test_file_format.py` | The Python wrapper `Shuttle`, error handling, `.twc` files |
| `test_learner.py`, `test_package.py` | `Learner`, solver and `Verifier` on a tiny problem; imports, registry, utilities |
| `test_e2e_spiral.py` | End-to-end (marked `e2e`): training on the spiral problem and solving the inverse problem |

### Continuous integration

For every push to `main` and every pull request, GitHub Actions ([.github/workflows/tests.yml](.github/workflows/tests.yml)) runs three jobs:

- **Tests:** on Python 3.10 to 3.14, with the CPU-only package.
- **CUDA build:** compiles the CUDA backend in a container with the CUDA toolkit.
- **Source distribution:** builds the source distribution and tests the package installed from it.

The runners of GitHub have no GPU, so the GPU tests are not run there. Please run the tests on a machine with a GPU before submitting changes of the C++/CUDA core.

The workflow [.github/workflows/release.yml](.github/workflows/release.yml) publishes releases, see [Releasing](#releasing).

### Documentation

- [README.md](README.md) is the main documentation, shown on GitHub. Formulas are written as ```` ```math ```` blocks and inline as `` $`...`$ ``: GitHub passes these to MathJax unchanged, whereas `$...$` and `$$...$$` are subject to Markdown processing (backslashes and underscores are lost).
- [docs/pypi_description.md](docs/pypi_description.md) is the text of the package page on PyPI, which cannot render formulas and the images of the README. Its example is the same as in the section "Getting started" of the README; please keep them in sync.
- [CHANGELOG.md](CHANGELOG.md) lists notable changes for each release.

### Plain CMake build (C++ benchmarks)

With the virtual environment activated:

```bash
cmake -S . -B build/cmake -G Ninja -DTWINCHER_BUILD_BENCHMARKS=ON
cmake --build build/cmake
build/cmake/bench_cpu 0 6    # mode 0 (inference), 6 threads
```

### Build options

The following CMake options can be passed to pip as `-Ccmake.define.<OPTION>=<VALUE>`:

| Option | Default | Meaning |
|---|---|---|
| `TWINCHER_CUDA` | `AUTO` | GPU backend. `AUTO`: build it if a CUDA compiler (nvcc) is found, otherwise build CPU-only with a warning. `ON`: require it. `OFF`: disable it. |
| `CMAKE_CUDA_ARCHITECTURES` | `native` | GPU architectures to compile for, e.g. `"86;120"`. The environment variable `CUDAARCHS` can be used instead. |
| `TWINCHER_NATIVE_ARCH` | `OFF` | Optimize host code for the CPU of the build machine (`-march=native`). The result is not portable. |
| `TWINCHER_BUILD_BENCHMARKS` | `OFF` | Build the C++ benchmark executables. |

### Building distributions

```bash
pip install build
python -m build    # creates dist/twincher-<version>.tar.gz and a wheel
```

## Releasing

Releases are published on PyPI as source distributions by the workflow [.github/workflows/release.yml](.github/workflows/release.yml), using trusted publishing (no API tokens).

### For each release

1. Make sure that the tests pass on GitHub (workflow *Tests*) and locally on a machine with a GPU (`python -m pytest`).
2. Set the new version in `pyproject.toml` and `CITATION.cff` (also `date-released`), and move the changes in `CHANGELOG.md` under a heading with the version and date. Commit and push to `main`.
3. Optionally, make a trial run on TestPyPI: in the *Actions* tab, run the workflow *Release* manually. Then install from TestPyPI in a new environment:

   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ twincher
   ```

   TestPyPI, like PyPI, accepts each version only once; for repeated trials use pre-release versions such as `0.2.0rc1`.
4. Create a release on GitHub with the tag `v<version>` (for example `v0.1.0`), with the changes from `CHANGELOG.md` as description. Publishing the release starts the upload to PyPI.

## Reporting issues

When reporting a bug, please include:

- the Twincher version (`python -c "import twincher; print(twincher.__version__)"`);
- your operating system, Python version, and CUDA version (if relevant);
- a minimal script that reproduces the problem.
