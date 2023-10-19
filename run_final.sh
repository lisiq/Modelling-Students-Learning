#!/bin/sh
papermill -p IRT_DIMS 1 -p TEST False GNN_final.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/GNN_batch_IRT.html > logs/GNN_batch_IRT.log &
papermill -p IRT_DIMS 0 -p TEST False GNN_final.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/GNN_batch_SAGE.html > logs/GNN_batch_IRT.log