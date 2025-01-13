#!/bin/sh
NPERMS=1000

## full

# with scales
papermill -p DATASET full -p OUTNAME SAGE_scales -p NPERMS $NPERMS Plot_results.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/Plot_results_full_scales.html > logs/Plot_results_full_scales.log 

# without scales
#papermill -p DATASET full -p OUTNAME SAGE -p NPERMS $NPERMS Plot_results.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/Plot_results_full.html > logs/Plot_results_full.log 

## matrix and topic
papermill -p DATASET matrix -p OUTNAME SAGE_scales -p NPERMS $NPERMS Plot_results.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/Plot_results_matrix_scales.html > logs/Plot_results_matrix_scales.log 
#papermill -p DATASET topic -p OUTNAME SAGE_scales -p NPERMS $NPERMS Plot_results.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/Plot_results_topic_scales.html > logs/Plot_results_topic_scales.log

# with scales and RT
#papermill -p DATASET full -p OUTNAME SAGE_RT_scales -p NPERMS $NPERMS Plot_results.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/Plot_results_full_RT_scales.html > logs/Plot_results_RT_full_scales.log 

#Rscript ArrangePlots.R