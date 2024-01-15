#!/bin/sh
papermill -p IRT_DIMS 1 -p DATASET full GNN_final.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/GNN_batch_IRT_full.html > logs/GNN_batch_IRT_full.log 
papermill -p IRT_DIMS 0 -p DATASET full GNN_final.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/GNN_batch_SAGE_full.html > logs/GNN_batch_SAGE_full.log
papermill -p IRT_DIMS 0 -p DATASET matrix -p ITEM_FEATURES True GNN_final.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/GNN_batch_SAGE_matrix_scales.html > logs/GNN_batch_SAGE_matrix.log
papermill -p IRT_DIMS 0 -p DATASET topic -p ITEM_FEATURES True GNN_final.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/GNN_batch_SAGE_topic_scales.html > logs/GNN_batch_SAGE_topic.log

papermill -p IRT_DIMS 0 -p DATASET full -p ITEM_FEATURES True GNN_final.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/GNN_batch_SAGE_full_scales.html > logs/GNN_batch_SAGE_full_scales.log

./run_plots.sh