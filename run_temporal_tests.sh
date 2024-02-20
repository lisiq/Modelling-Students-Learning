#!/bin/sh

## matrix and topic
# with scales
papermill -p IRT_DIMS 0 -p DATASET matrix -p ITEM_FEATURES True GNN_final.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/GNN_batch_SAGE_matrix_scales.html > logs/GNN_batch_SAGE_matrix_scales.log
papermill -p IRT_DIMS 1 -p DATASET matrix -p ITEM_FEATURES True GNN_final.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/GNN_batch_IRT_matrix_scales.html > logs/GNN_batch_IRT_matrix_scales.log 