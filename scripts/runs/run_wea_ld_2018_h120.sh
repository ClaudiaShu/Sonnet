#!/bin/bash
python scripts/run_experiment.py \
    model=sonnet \
    dataset=weatherbench/london/london_2018 \
    exp=weatherbench \
    exp.batch_size=64 \
    exp.epochs=100 \
    exp.learning_rate=0.0002 \
    exp.pred_length=120 \
    exp.seq_length=28 \
    model.model_params.d_model=32 \
    model.model_params.n_atoms=16
