#!/bin/bash

# Script that runs all cbc.block demos and exits if anything goes wrong. If
# this script finishes successfully, we are reasonably confident that there are
# no api-related breakages (and other types of error will often lead to a lack
# of convergence, so in practice they may be apparent too).

set -e
export DOLFIN_NOPLOT=1

cd ${0%/*}
demos=$(find . -name \*.py)

if which parallel &>/dev/null; then
    parallel --gnu -j +0 --halt-on-error=2 -v -n 1 python ::: $demos
else
    for demo in $demos; do
	echo python $demo
	python $demo
    done
fi

for demo in $demos; do
    echo mpirun -np 3 python $demo
    mpirun -np 3 python $demo
done

ps -o etime,cputime $$
