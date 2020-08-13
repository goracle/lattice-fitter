#!/bin/bash

mkdir tocut
combine_fit_results.sh && \
    hist.py $(ls *cp | grep phase | grep -v err) 2>&1
final_fits.py
