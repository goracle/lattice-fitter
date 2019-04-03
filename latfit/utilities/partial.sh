#!/bin/bash

echo "assuming first argument is destination"
echo "assuming second argument is the partial directory to combine with the destination"
if [ ! -d $1 ]; then
    echo "bad first input"
    exit 1
fi

if [ ! -d $2 ]; then
    echo "bad second input"
    exit 1
fi
if [ ! "$(basename $1)" = "$(basename $2)" ]; then
    echo "trajectories are not the same"
    exit 1
fi
testd () {
    DIR=$1
    if [ ! "${DIR:0:1}" = "/" ]; then
	rel=1
    else
	rel=0
    fi
    echo $rel
}

rel1=$(testd $1)
rel2=$(testd $2)
if [ $rel1 -eq 0 ]; then
    dir1=$1
else
    dir1=$(pwd)/$1
fi
if [ $rel2 -eq 0 ]; then
    dir2=$2
else
    dir2=$(pwd)/$2
fi

cd $dir2/props/output
for file in *hdf5
do
    newname=$(echo $file | sed 's/.hdf5/000&/')
    h5diff -q $file $dir1/props/output/$file
    if [ $? -ne 0 ]; then
	echo "linking $file $dir1/props/output/$newname"
	ln -s $(pwd)/$file $dir1/props/output/$newname
    else
	echo "files $file , $dir1/props/output/$file"
	echo "are identical.  Not linking."
	continue
    fi
done
echo "finished linking all files to $dir1"
