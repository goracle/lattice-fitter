#!/bin/bash

#quantity_of_interest=$1
combine_pickle.py $(ls energy*.p | grep -v err)
echo "finished: energy"
combine_pickle.py $(ls energy*.p | grep err)
echo "finished: energy err"
combine_pickle.py $(ls pvalue*.p | grep err)
echo "finished: pvalue err"
combine_pickle.py $(ls pvalue*.p | grep -v err)
echo "finished: pvalue"
combine_pickle.py $(ls phase*.p | grep -v err)
echo "finished: phase shift"
combine_pickle.py $(ls phase*.p | grep err)
echo "finished: phase shift err"
