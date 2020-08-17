# NBA Daily Fantasy Sports

Predicting NBA daily fantasy scores.

## Setup

Create python environment:

```bash
conda env create -f environment.yml -n nbadfs
conda activate nbadfs
```

## Notebooks

- [0-initial-data-load.ipynb](./0-initial-data-load.ipynb): Load historical season, player, boxscore, lineup and daily fantasy salary information from mysportsfeeds into a mongo database
- [1-daily-data-update.ipynb](./1-daily-data-update.ipynb): Make incremental daily updates to input datasets, loaded from mysportsfeeds
- [2-lineup-predictor.ipynb](./2-lineup-predictor.ipynb): Create an estimate of today's NBA lineups with additional markers for sense-checking or manually overriding projected lineups
- [3-feature-engineering.ipynb](./3-feature-engineering.ipynb): Generate features used in daily fantasy predictive model
- [4-dfs-model-build.ipynb](./4-dfs-model-build.ipynb): Build a model to predict daily fantasy scores and generate a lineup for upcoming games based on the model
