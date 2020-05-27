# Uncertainty Forest

This repo contains code and demos of estimation procedures for posterior distributions, conditional entropy, and mutual information between random variables `X` and `Y`.

- [Overview](#overview)
- [System Requirements](#system-requirements)

# Overview
To reproduce any of the figures, navigate to the corresponding directory, and run the Jupyter notebook.
```
cd figs/fig1
jupyter nbconvert --to notebook --inplace --execute figure-1.ipynb --ExecutePreprocessor.timeout=-1
```
Commands are similar for Figures 2 and 3. The application and hypothesis test code can be found in the `figs/application` director. The above commands convert the notebook to a Python file and produces the figures as PDFs. An alternate option is to open to the notebook, and select "Restart and Run All".

# System Requirements
## Hardware requirements
UF requires only a standard computer with enough RAM to support the in-memory operations.

## Software requirements
### OS Requirements
The code has been tested on the following systems:
+ macOS: Mojave (10.14.1)
+ Linux: Ubuntu 16.04

### Python Dependencies
The code mainly depends on the Python scientific stack.

```
numpy
scipy
scikit-learn
joblib
matplotlib
seaborn
tqdm
```
