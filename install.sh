#!/bin/bash

# install.sh

conda create -y -n se_env python=3.7
conda activate se_env

conda install -y -c pytorch pytorch=1.4.0 cudatoolkit=10.0
conda install -y scikit-learn=0.22.1

pip install tqdm
pip install graspy
pip install networkx
pip install git+https://github.com/bkj/ez_ppnp.git
