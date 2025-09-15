# Contributing to LinAlgKit

Thanks for your interest in contributing! This document outlines how to set up your environment, propose changes, and follow our guidelines.

## Development setup

- Requires: Python 3.8+, C++17 compiler, CMake 3.10+, Git
- Recommended: Linux/WSL for building wheels and running benchmarks

### Quickstart

```bash
# Clone your fork
git clone https://github.com/<your-username>/LinAlgKit.git
cd LinAlgKit

# Python editable install
pip install -U pip scikit-build-core pybind11 numpy
pip install -e .

# Build and run C++ tests
mkdir -p build && cd build
cmake -DBUILD_TESTS=ON ..
cmake --build . -j
ctest --output-on-failure
```

## Code style & guidelines

- C++: Prefer modern C++17 features, clear naming, and bounds checks.
- Python: PEP8 style, type hints where appropriate.
- Tests: Add/extend unit tests (C++ GTest) for new features and bugfixes.
- Benchmarks: For performance-sensitive changes, add or extend benchmarks.

## Pull Requests

- Open an issue first for significant changes.
- Keep PRs focused and reasonably sized.
- Include tests and update docs where relevant.
- Ensure CI is green.

## Commit Messages

- Use meaningful subject lines (max ~72 chars).
- Reference issues (e.g., `Fixes #123`).

## Reporting Bugs

- Use the `Bug report` issue template.
- Include environment info (OS, Python, compiler, version).
- Provide minimal reproduction steps.

## Feature Requests

- Use the `Feature request` issue template.
- Clearly describe the problem and desired solution.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
