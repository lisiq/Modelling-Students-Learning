#!/bin/sh
papermill -p DATASET full Plot_results.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/Plot_results_full.html > logs/Plot_results_full.log 
papermill -p DATASET matrix Plot_results.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/Plot_results_matrix.html > logs/Plot_results_matrix.log 
papermill -p DATASET topic Plot_results.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/Plot_results_topic.html > logs/Plot_results_topic.log 

Rscript ArrangePlots.R
