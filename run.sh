#!/bin/bash

# run.sh

# --
# Setup environment

conda create -y -n se_env python=3.7
conda activate se_env

conda install -y -c pytorch pytorch cudatoolkit=10.0
conda install -y scikit-learn

pip install tqdm
pip install graspy
pip install networkx
pip install git+https://github.com/bkj/rsub
pip install git@https://github.com/bkj/ez_ppnp.git

# --
# Generic

python main.py --inpath small_data/cora/cora | jq .
python main.py --inpath small_data/citeseer/citeseer --hidden-dim 32 | jq .
python main.py --inpath small_data/pubmed/pubmed | jq .




