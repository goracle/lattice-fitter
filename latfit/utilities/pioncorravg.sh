#!/bin/bash

p111=$(ls | grep pioncorr | grep -v 'unsummed' | grep -v 0 | grep jkdat | grep -v mom1.j | grep -v mom11.jk)
p1=$(ls | grep pioncorr | grep -v 000 | grep -v 'unsummed' | grep 0| grep -v 11 | grep -v 1_1 | grep -v 10_1 | grep -v 101)
p11=$(ls | grep pioncorr | grep -v 00 | grep -v 010 | grep -v 'unsummed' | grep -v 0_10 | grep 0 | grep jkdat);
echo "averaging p111:"
echo ""
echo "pioncorrChk_p111"
echo ""
echo $p111
avg_hdf5.py $p111
echo "averaging p11:"
echo ""
echo "pioncorrChk_p11"
echo ""
echo $p11
avg_hdf5.py $p11
echo "averaging p1:"
echo ""
echo "pioncorrChk_p1"
echo ""
echo $p1
avg_hdf5.py $p1
