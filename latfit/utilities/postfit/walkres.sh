#!/bin/bash

a=$(pwd)

for i in $@
do
    cd $i || exit 1
    i=$(echo $i | tr -s '_' | tr '_' ' ')
    echo "# $i"
    grep -A2 directed $(slat.sh) | tail -1
    cd $a
done

arr=()
for i in $@
do
    cd $i || exit 1
    b=$(grep -A2 directed $(slat.sh) | tail -1)
    #arr+=(\"$b\")
    arr+=($b)
    arr+=("@")
    cd $a
done
echo ""
walkpy.py "${arr[@]}"
#echo "${arr[@]}"
