# SD-KDE: Score-Debiased Kernel Density Estimation

# About
This repository contains the code for the paper "Score-Debiased Kernel Density Estimation".

## Requirements
- Python 3.9
- NumPy 1.20
- Matplotlib 3.3
- Scipy 1.7

## Installation
You can set up the environment using conda:
```
conda create -n sd-kde numpy scipy matplotlib
```

## Experiments

### Gaussian Mixture 1D Experiments
To generate Figure 2, Figure 3, Figure 4, Figure 8, run the following command:
```
python shrinkage_kde_gaussian.py
```

### Laplace Mixture 1D Experiments
To generate Figures 9, Figure 10, Figure 11, run the following command:
```
python shrinkage_kde_laplace.py
```
### Score Visualization
To generate Figure 11 and Figure 12, run the following command:
```
python visualize_score.py
```


