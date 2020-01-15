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
# Run JHU on maggot

python jhu_vn_maggot.py --mode ase
# mode=ase
# f1.mean [0.86123523 0.94151383 0.28093112 0.54028885]
# f1.std [0.04481212 0.01754654 0.19963374 0.20195913]

python jhu_vn_maggot.py --mode lse
# f1.mean [0.81673811 0.79135892 0.29380187 0.75851432]
# f1.std [0.06263664 0.14081764 0.19825613 0.18505287]


# --
# Desikan

# prep
python prep_desikan.py

# run PPNP
python ppnp_desikan.py \
    --ppr-outpath ./data/DS72784/subj1-scan1.ppr_array.npy
# {'acc': 0.563650223238593, 'f1_macro': 0.5203338970280309, 'f1_micro': 0.563650223238593}

# run ASE
python jhu_vn_desikan.py
