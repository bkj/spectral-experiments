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
# Download data

# maggot
# ... download from https://github.com/neurodata/maggot_models/tree/master/data/processed/mb_2019-09-23 ...

# DS72784
mkdir -p data/DS72784

wget http://www.cis.jhu.edu/~parky/TT/Data/DS72784/Labels.zip \
    -O data/DS72784/Labels.zip

wget http://www.cis.jhu.edu/~parky/TT/Data/DS72784/Graphs/DS72784-graphml-raw.zip \
    -O data/DS72784/DS72784-graphml-raw.zip

cd data/DS72784
unzip Labels.zip
unzip DS72784-graphml-raw.zip

mv Labels/* DS72784/* ./
rm -r Labels DS72784
rm Labels.zip
rm DS72784-graphml-raw.zip


# --
# Maggot

python jhu_vn_maggot.py --mode ase
# mode=ase
# f1.mean [0.86123523 0.94151383 0.28093112 0.54028885]
# f1.std [0.04481212 0.01754654 0.19963374 0.20195913]

python jhu_vn_maggot.py --mode lse
# f1.mean [0.81673811 0.79135892 0.29380187 0.75851432]
# f1.std [0.06263664 0.14081764 0.19825613 0.18505287]

# ppnp -- skipping for right now.

# --
# Desikan

# prep
python prep_desikan.py

# run PPNP
python ppnp_desikan.py \
    --ppr-alpha 0.1    \
    --ppr-outpath ./data/DS72784/subj1-scan1.ppr_array0.1.npy

# {'acc': 0.5895132309702712, 'f1_macro': 0.5445337921270678, 'f1_micro': 0.5895132309702712}

# run ASE
python jhu_vn_desikan.py --seed 111
# {'acc': 0.4854350430142655, 'f1_macro': 0.4300804126303814, 'f1_micro': 0.4854350430142655}

# --
# CORA

python ase_generic.py --inpath data/cora/cora
# {'method': 'ase', 'acc': 0.5085013839462238, 'f1_macro': 0.45474561716262213, 'f1_micro': 0.5085013839462238}
python ppnp_generic.py --inpath data/cora/cora
# {'method': 'ppnp', 'acc': 0.7900355871886121, 'f1_macro': 0.7483827681960851, 'f1_micro': 0.7900355871886121}

python ase_generic.py --inpath data/citeseer/citeseer
# {'method': 'ase', 'acc': 0.47972617166929965, 'f1_macro': 0.37950913292634175, 'f1_micro': 0.47972617166929965}
python ppnp_generic.py --inpath data/citeseer/citeseer
# {'method': 'ppnp', 'acc': 0.6924697209057399, 'f1_macro': 0.6259157269425071, 'f1_micro': 0.6924697209057399}

python ase_generic.py --inpath data/pubmed/pubmed
# {'method': 'ase', 'acc': 0.7556632480558999, 'f1_macro': 0.7321985772004179, 'f1_micro': 0.7556632480558999}
python ppnp_generic.py --inpath data/pubmed/pubmed
# {'method': 'ppnp', 'acc': 0.765242871633044, 'f1_macro': 0.7476222554914792, 'f1_micro': 0.765242871633044}


# ..

python generic.py --inpath data/cora/cora | jq .
# {
#   "dataset": "data/cora/cora",
#   "n_nodes": 2810,
#   "n_edges": 15962,
#   "ppnp_scores": {
#     "acc": 0.7900355871886121,
#     "f1_macro": 0.7483827681960851,
#     "f1_micro": 0.7900355871886121
#   },
#   "ase_scores": {
#     "acc": 0.7346777382364571,
#     "f1_macro": 0.6918257994053247,
#     "f1_micro": 0.7346777382364571
#   }
# }

python generic.py --inpath data/citeseer/citeseer | jq .


