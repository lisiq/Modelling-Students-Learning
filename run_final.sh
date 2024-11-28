#!/bin/sh
# experiments
#  nohup python run_experiments_zh.py > experiments.log &

## full SAGE
# without scales
#papermill -p IRT_DIMS 0 -p DATASET full -p ITEM_FEATURES False GNN_final.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/GNN_batch_SAGE_full.html > logs/GNN_batch_SAGE_full.log
# with scales
papermill -p IRT_DIMS 0 -p DATASET full -p ITEM_FEATURES True GNN_final.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/GNN_batch_SAGE_full_scales.html > logs/GNN_batch_SAGE_full_scales.log

## full IRT
# without scales
#papermill -p IRT_DIMS 1 -p DATASET full -p ITEM_FEATURES False GNN_final.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/GNN_batch_IRT_full.html > logs/GNN_batch_IRT_full.log
#rm results/IRT_full_aux_data.pkl
# with scales
papermill -p IRT_DIMS 1 -p DATASET full -p ITEM_FEATURES True GNN_final.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/GNN_batch_IRT_full_scales.html > logs/GNN_batch_IRT_full_scales.log 
rm results/IRT_scales_full_aux_data.pkl

## matrix and topic
# without scales
#papermill -p IRT_DIMS 0 -p DATASET matrix -p ITEM_FEATURES False GNN_final.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/GNN_batch_SAGE_matrix.html > logs/GNN_batch_SAGE_matrix.log

# with scales
#papermill -p IRT_DIMS 0 -p DATASET matrix -p ITEM_FEATURES True GNN_final.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/GNN_batch_SAGE_matrix_scales.html > logs/GNN_batch_SAGE_matrix_scales.log
#papermill -p IRT_DIMS 0 -p DATASET topic -p ITEM_FEATURES True GNN_final.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/GNN_batch_SAGE_topic_scales.html > logs/GNN_batch_SAGE_topic_scales.log

## matrix and topic IRT
#papermill -p IRT_DIMS 1 -p DATASET matrix -p ITEM_FEATURES False GNN_final.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/GNN_batch_IRT_matrix.html > logs/GNN_batch_IRT_matrix.log 
#papermill -p IRT_DIMS 1 -p DATASET topic -p ITEM_FEATURES False GNN_final.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/GNN_batch_IRT_topic.html > logs/GNN_batch_IRT_topic.log 

./run_plots.sh