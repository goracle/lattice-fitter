#!/bin/bash

for file in $(ls | egrep -o '^[0-9]+$');
do
    if [ -d "$(pwd)/job-0${file}" ]; then
	echo "File job-0${file} already exists; continuing"
	continue
    fi
    ln -s $file job-0${file};
done
a=$(pwd)
for file in $(ls | grep 'job-0');
do
    b=$(echo $file | sed 's/job-0//');
    if [ -f "${a}/traj_${b}.hdf5" ]; then
	echo "File $traj_${b}.hdf5 already exists; continuing"
	continue
    fi
    wd=$(pwd)
    echo "working dir=${wd}"
    cd $file/props/output || exit 1;
    rm -f traj_${b}.hdf5;
    h5link.py || exit 1;
    ln -s "$(pwd)/traj_${b}.hdf5" "$a/traj_${b}.hdf5";
    cd ../../..;
done

