#!/bin/bash

a=$(pwd)

for i in $@
do
    cd $i || exit 1
    i=$(echo $i | tr -s '_' | tr '_' ' ')
    ls proc_params.p > /dev/null || exit 1
    cuts=$(proc_params.py | grep -c "cuts': False")
    if [ $cuts -eq 1 ]; then
	echo "# $i, weak cuts"
    else
	echo "# $i, strong cuts"
    fi
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
    cuts=$(proc_params.py | grep -c "cuts': False")
    if [ $cuts -eq 1 ]; then
	arr+=("B")
    else
	arr+=("@")
    fi
    cd $a
done
echo ""
walkpy.py "${arr[@]}"
#echo "${arr[@]}"
