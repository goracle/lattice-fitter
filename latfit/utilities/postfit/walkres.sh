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

    # two different ways to get energy phase pairs
    # first uses independent walkbacks
    # grep -A2 directed $(slat.sh) | tail -1

    # second uses one walkback on the energy, then finds
    # the corresponding phase shift calculated from that energy
    paired_energies_phases.py $(slat.sh) 'False'
    cd $a
done
arr=()
for i in $@
do
    cd $i || exit 1

    # two different ways to get energy phase pairs
    # first uses independent walkbacks
    #b=$(#paired_energies_phases.py $(#slat.sh));
    b=$(grep -A2 directed $(slat.sh) | tail -1)

    # second uses one walkback on the energy, then finds
    # the corresponding phase shift calculated from that energy
    b=$(paired_energies_phases.py $(slat.sh) 'True')

    if [ "$b" = "[]" ]; then
	cd $a
	continue
    fi

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
