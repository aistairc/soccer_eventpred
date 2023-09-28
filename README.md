# Soccer Event Prediction

## Introduction

This repository contains code for the paper "Leveraging Player Embeddings for Soccer Event Prediction".

## Getting Started

### Environment

Type `make setup` to create a virtual environment and install all dependencies.

### Data

Raw data provided by wyscout is supposed to be placed in `data/wyscout` directory.
The directory structure should be as follows:

```
data/wyscout
└── raw
    ├── events
    │   ├── events_England.json
    │   ├── events_France.json
    │   ├── events_Germany.json
    │   ├── events_Italy.json
    │   ├── events_Spain.json
    │   └── events_World_Cup.json
    └──mappings
        ├── players.json
        ├── tags2name.csv
        └── teams.json
```

A large part of the event preprocessing code is borrowed from [seq2Event](https://github.com/statsonthecloud/Soccer-SEQ2Event).

```
poetry run python scripts/preprocess_data.py teams \
    --input_path data/wyscout_offense_only/raw/mappings/teams.json \
    --output_path data/wyscout_offense_only/processed/mappings/id2team.json

poetry run python scripts/preprocess_data.py players \
    --input_path data/wyscout_offense_only/raw/mappings/players.json \
    --output_path data/wyscout_offense_only/processed/mappings/id2player.json

poetry run python scripts/preprocess_data.py tags \
    --input_path data/wyscout_offense_only/raw/mappings/tags2name.csv \
    --output_path data/wyscout_offense_only/processed/mappings/tagid2name.json

poetry run python scripts/preprocess_data.py events \
    --input_path data/wyscout_offense_only/raw/events \
    --output_dir data/wyscout_offense_only/processed/events \
    --mappings_dir data/wyscout_offense_only/processed/mappings \
    --targets "events_Spain.json" \
    --offense_only true

poetry run python scripts/preprocess_data.py split \
    --df_pickle_path data/wyscout_offense_only/processed/events/all_preprocessed.pkl \
    --output_dir data/wyscout_offense_only/processed/events \
    --random_state 42
```

### Modeling

```
python scripts/train.py \
    --data-name "wyscout_offense_only" \
    --config configs/sample_model.jsonnet \
    --mapping configs/label2events_seq2event_offense_only.json \
    --exp-name "test" \
    --name "test_run" \
    --epochs 1 \
    --gradient-accumulation-steps 1 \
    --class-weight-type "exponential" \
    --beta 0.9 \
    --accelerator "gpu" \
    --devices 1
```

## Evaluation

```
python scripts/evaluate.py \
    --data-name "wyscout_offense_only" \
    --config configs/sample_model.jsonnet \
    --mapping configs/label2events_seq2event_offense_only.json \
    --run-name "test_run" \
    --class-weight-type "exponential" \
    --beta 0.9

python scripts/confusion_matrix.py \
    --run-name "test_run"
```
