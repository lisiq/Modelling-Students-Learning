#!/bin/sh

## matrix and topic
# without scales
papermill -p IRT_DIMS 0 -p DATASET matrix -p ITEM_FEATURES False GNN_final.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/GNN_batch_SAGE_matrix.html > logs/GNN_batch_SAGE_matrix.log
# with scales
papermill -p IRT_DIMS 0 -p DATASET matrix -p ITEM_FEATURES True GNN_final.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/GNN_batch_SAGE_matrix_scales.html > logs/GNN_batch_SAGE_matrix_scales.log
papermill -p IRT_DIMS 0 -p DATASET topic -p ITEM_FEATURES True GNN_final.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/GNN_batch_SAGE_topic_scales.html > logs/GNN_batch_SAGE_topic_scales.log

## full SAGE
# without scales
papermill -p IRT_DIMS 0 -p DATASET full -p ITEM_FEATURES False GNN_final.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/GNN_batch_SAGE_full.html > logs/GNN_batch_SAGE_full.log
# with scales
papermill -p IRT_DIMS 0 -p DATASET full -p ITEM_FEATURES True GNN_final.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/GNN_batch_SAGE_full_scales.html > logs/GNN_batch_SAGE_full_scales.log

## full IRT
papermill -p IRT_DIMS 1 -p DATASET full -p ITEM_FEATURES False GNN_final.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/GNN_batch_IRT_full.html > logs/GNN_batch_IRT_full.log 

./run_plots.sh