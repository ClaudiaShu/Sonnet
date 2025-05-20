#!/bin/bash
python scripts/run_experiment.py \
    model=sonnet \
    dataset=weatherbench/hongkong/hongkong_2016 \
    exp=weatherbench \
    exp.batch_size=64 \
    exp.epochs=100 \
    exp.learning_rate=0.005 \
    exp.pred_length=120 \
    exp.seq_length=28 \
    model.model_params.d_model=8 \
    model.model_params.n_atoms=64
