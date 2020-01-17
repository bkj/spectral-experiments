# spectral-experiments

## Installation

See `./install.sh`

## Usage

See `./run.sh`

## Results

See `./results/*.json`

In this setup:
 - On `{cora, citeseer, pubmed}`, `ppr_full` is the best unsupervised method.  
 - `ppnp_supervised` wins on `cora`, but not `{citeseer, pubmed}`.  I'd guess this is an overfitting problem, and `ppnp_supervised` would dominate if we add regularization (specifically early stopping)

## Notes

- How to handle directed graphs?  How to handle weighted graphs?  (See `--no-sym` and `--no-bin` flags.)
- Not using node features in any of these experiments
- `RandomForestClassifier` may not always be the best model for downstream tasks