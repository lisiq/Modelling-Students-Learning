#!/bin/sh
papermill -p DATASET full Plot_results.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/Plot_results_full.html > logs/Plot_results_full.log 

papermill -p DATASET full -p SAGE_scales Plot_results.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/Plot_results_full_scales.html > logs/Plot_results_full_scales.log 
papermill -p DATASET matrix -p SAGE_scales Plot_results.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/Plot_results_matrix_scales.html > logs/Plot_results_matrix_scales.log 
papermill -p DATASET topic -p SAGE_scales Plot_results.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/Plot_results_topic_scales.html > logs/Plot_results_topic_scales.log

Rscript ArrangePlots.R
