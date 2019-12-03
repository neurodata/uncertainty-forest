# uncertainty_forest

`uncertainty_forest` is a Python package containing estimation procedures for posterior distributions, conditional entropy, and mutual information between random variables `X` and `Y`.

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [License](#license)
- [Issues](https://github.com/neurodata/uncertainty-forest/issues)

# Overview
See paper: https://arxiv.org/abs/1907.00325

# System Requirements
## Hardware requirements
`uncertainty_forest` package requires only a standard computer with enough RAM to support the in-memory operations.

## Software requirements
### OS Requirements
This package is supported for standard operating systems: *macOS*, *Windows*, and *Linux*. The package has been tested on the following systems:
+ macOS: Mojave (10.14.1)
+ Linux: Ubuntu 16.04

### Python Dependencies
`uncertainty_forest` mainly depends on the Python scientific stack.

```
numpy
scipy
scikit-learn
joblib
```

# Installation Guide:

### Install from Github
```
git clone https://github.com/neurodata/uncertainty-forest
cd uncertainty-forest
python3 setup.py develop
```

# License

This project is covered under the **Apache 2.0 License**.