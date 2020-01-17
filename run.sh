#!/bin/bash

# run.sh

# --
# Canonical datasets

python main.py --inpath small_data/cora/cora --hidden-dim 16 | jq .

python main.py --inpath small_data/citeseer/citeseer --hidden-dim 16 | jq .

python main.py --inpath small_data/pubmed/pubmed

# --
# Desikan

python datasets/desikan.py

