#!/bin/bash

#quantity_of_interest=$1
combine_pickle.py $(ls energy*.p | grep -v err)
combine_pickle.py $(ls energy*.p | grep err)
combine_pickle.py $(ls pvalue*.p | grep err)
combine_pickle.py $(ls pvalue*.p | grep -v err)
combine_pickle.py $(ls phase*.p | grep -v err)
combine_pickle.py $(ls phase*.p | grep err)
