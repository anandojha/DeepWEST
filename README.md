mdnet
==============================

Machine Learning Analysis of Molecular Dynamics Trajectories for Weighted Ensemble simulations


## Installation and Setup Instructions :
* Make sure [anaconda3](https://www.anaconda.com/) is installed on the local machine. Go to the  [download](https://www.anaconda.com/products/individual) page of anaconda3 and install the latest version of anaconda3.
* Create a new conda environment with python = 3.7 and install the package with the following commands in the terminal: 
```bash
conda create -n mdnet python=3.7

conda activate mdnet

conda install -c conda-forge matplotlib scipy jupyterlab pandas tensorflow mdtraj scikit-learn seaborn

