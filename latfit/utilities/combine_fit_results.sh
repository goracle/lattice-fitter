#!/bin/bash

#quantity_of_interest=$1
a=0
prl.sh tocut/* .
while [ $a -eq 0 ];
do
    combine_pickle.py $(ls energy*.p | grep -v err)
    a=$?
    if [ ! $a -eq 0 ]; then break; fi
    echo "finished: energy"
    combine_pickle.py $(ls energy*.p | grep err)
    a=$?
    if [ ! $a -eq 0 ]; then break; fi
    echo "finished: energy err"
    combine_pickle.py $(ls pvalue*.p | grep err)
    a=$?
    if [ ! $a -eq 0 ]; then break; fi
    echo "finished: pvalue err"
    combine_pickle.py $(ls pvalue*.p | grep -v err)
    a=$?
    if [ ! $a -eq 0 ]; then break; fi
    echo "finished: pvalue"
    combine_pickle.py $(ls phase*.p | grep -v err)
    a=$?
    if [ ! $a -eq 0 ]; then break; fi
    echo "finished: phase shift"
    # important that this be last
    combine_pickle.py $(ls phase*.p | grep err)
    a=$?
    if [ ! $a -eq 0 ]; then break; fi
    echo "finished: phase shift err"
done
echo "no more files found; exiting."
prl.sh tocut/* .
