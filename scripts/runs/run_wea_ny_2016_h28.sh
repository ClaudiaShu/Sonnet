#!/bin/bash
python scripts/run_experiment.py \
    model=sonnet \
    dataset=weatherbench/newyork/newyork_2016 \
    exp=weatherbench \
    exp.batch_size=64 \
    exp.epochs=100 \
    exp.learning_rate=0.0002 \
    exp.pred_length=28 \
    exp.seq_length=28 \
    model.model_params.d_model=32 \
    model.model_params.n_atoms=16
