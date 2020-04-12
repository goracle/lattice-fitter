#!/bin/bash

# eliminate fit results ruled out by a race condition.
if [ ! -d raced ]; then
    mkdir raced
else
    [ "$(ls -A raced)" ] && exit 1 || echo "raced is empty"
fi
for file in badfit_*;
do
    a=$(echo $file | sed 's/badfit//')
    if [ -f "pvalue$a" ]; then
	echo "moving $a to raced"
	prl.sh *$a raced/
    fi
done
