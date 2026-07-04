# SEAT Release Candidate

This folder is a local release candidate for the SEAT code used in the TongueDx experiments.

## Contents

- `src/`: runnable flat copy of the selected Python scripts. The original code uses flat imports such as `from common import ...`, so this directory is the safest starting point for execution.
- `seat/`: the same selected scripts grouped by role for review.
- `scripts/`: selected shell scripts used for paper experiments and ablations.
- `results/`: small CSV/JSON result artifacts used to trace reported paper tables and baseline rows.

## Not Included

The following files were intentionally not pulled into this public-candidate folder:

- raw images and dataset files
- cached backbone features
- model checkpoints and weights
- large prediction caches
- server logs
- exploratory VLM scripts not needed for the SEAT paper release

## Notes Before Public Release

- Review hard-coded paths in `src/common.py` and experiment scripts.
- Replace private absolute paths with command-line arguments or documented environment variables.
- Add dataset access instructions instead of publishing raw data.
- Add license files after confirming dataset and model-license constraints.
- Run a clean reproduction pass from `src/` before publishing.
