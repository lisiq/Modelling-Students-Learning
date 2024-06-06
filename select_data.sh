#!/bin/sh
papermill -p NSUBS 10000 -p TEST False select_data.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/select_data.html
#papermill -p NSUBS 20000 -p TEST False select_data.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/select_data.html
#papermill -p NSUBS 20000 -p TEST True select_data.ipynb | jupyter-nbconvert --stdin --no-input --to html --output vis/select_data_test.html