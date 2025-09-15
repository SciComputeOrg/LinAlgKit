# Releasing LinAlgKit

This document describes how to cut a new release, publish wheels to TestPyPI or PyPI, and verify the artifacts.

## Versioning

- Bump the version in `pyproject.toml` under `[project] version`.
- Use tags of the form:
  - Pre-release (goes to TestPyPI): `vX.Y.ZrcN`, `vX.Y.ZaN`, `vX.Y.ZbN`, or `vX.Y.ZdevN`
  - Stable (goes to PyPI): `vX.Y.Z`

## Pre-release checklist

- Ensure CI is green on `main`.
- Update `README.md` and `CHANGELOG` (if present).
- Smoke test locally:
  ```bash
  # Build C++
  mkdir -p ~/matrixlib_build && cd ~/matrixlib_build
  cmake -G "Unix Makefiles" -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=OFF -DPYTHON_EXECUTABLE=$(which python3) /path/to/repo
  cmake --build . -j
  ctest --output-on-failure

  # Python editable install & smoke import
  pip install -U pip scikit-build-core pybind11 numpy
  pip install -e /path/to/repo
  python - <<'PY'
import LinAlgKit as lk
m = lk.Matrix(2,2,1.0)
print("OK:", m.to_numpy().shape)
PY
  ```

## TestPyPI publish (optional, recommended)

1) Build artifacts locally
```bash
python -m pip install -U build twine
python -m build
```

2) Upload to TestPyPI
```bash
# Create a token at https://test.pypi.org/account/ and set TWINE_PASSWORD env var
python -m twine upload -r testpypi dist/*
```

3) Verify install from TestPyPI in a clean venv
```bash
python -m venv .venv && source .venv/bin/activate
pip install -i https://test.pypi.org/simple/ LinAlgKit
python - <<'PY'
import LinAlgKit as lk
print(lk.Matrix(2,2,1.0).to_numpy())
PY
```

## GitHub Actions publish (recommended)

Publishing is automated via `.github/workflows/release.yml`:

- On pushing a tag `v*`, the workflow builds wheels (Linux/macOS/Windows) and an sdist.
- It uploads to TestPyPI for pre-release tags and PyPI for stable tags.

### Required repository secrets

- `PYPI_API_TOKEN`: PyPI token with upload permissions.
- `TEST_PYPI_API_TOKEN`: TestPyPI token with upload permissions.

### Triggering a release

```bash
git tag v0.1.0rc1   # pre-release -> TestPyPI
git push --tags
# or stable
git tag v0.1.0
git push --tags
```

## Post-release verification

- Install from the appropriate index:
```bash
# Stable from PyPI
pip install LinAlgKit
# Pre-release from TestPyPI
pip install -i https://test.pypi.org/simple/ LinAlgKit
```
- Run a smoke test import and a small operation.

## Troubleshooting

- Missing `cmake`/compiler during build-from-source:
  - Linux: `sudo apt-get install -y cmake build-essential python3-dev`
  - macOS: `xcode-select --install` (Command Line Tools)
  - Windows: Visual Studio Build Tools (Desktop C++ workload)
- NumPy headers not found:
  - `pip install -U numpy` before installing LinAlgKit
- Windows/WSL file locks:
  - Build in WSL `$HOME` and point CMake to Windows path instead of building on `/mnt/c`.
