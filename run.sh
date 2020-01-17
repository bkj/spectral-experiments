#!/bin/bash

# run.sh

# --
# Canonical datasets

python main.py --inpath small_data/cora/cora | jq . | tee results/cora.json

python main.py --inpath small_data/citeseer/citeseer | jq . | tee results/citeseer.json

python main.py --inpath small_data/pubmed/pubmed | jq . | tee results/pubmed.json

# --
# Desikan

python ./datasets/desikan.py --inpath ./data/DS72784/subj1-scan1 | jq .

python main.py                          \
    --inpath ./data/DS72784/subj1-scan1 \
    --se-components 8 | jq .