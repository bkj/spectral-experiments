#!/bin/bash

# run.sh

# --
# Setup environment

conda create -y -n se_env python=3.7
conda activate se_env

conda install -y scikit-learn

pip install tqdm
pip install graspy
pip install networkx

# --
# Run JHU

python jhu_vn.py