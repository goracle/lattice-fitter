#!/bin/bash

quantity_of_interest=$1
combine_pickle.py $(ls ${quantity_of_interest}*.p | grep -v err)
combine_pickle.py $(ls ${quantity_of_interest}*.p | grep err)
