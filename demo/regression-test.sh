#!/bin/bash

# Script that runs all cbc.block demos and exits if anything goes wrong. If
# this script finishes successfully, we are reasonably confident that there are
# no api-related breakages (and other types of error will often lead to a lack
# of convergence, so in practice they may be apparent too).

set -e
export DOLFIN_NOPLOT=1
export PYTHONPATH=$PWD/..:$PYTHONPATH

cd ${0%/*}
demos=$(find . -name \*.py)

if which parallel &>/dev/null; then
    parallel -j +0 --halt-on-error=2 -v -n 1 python ::: $demos
else
    for demo in $demos; do
	echo python $demo
	python $demo
    done
fi

# Only demos that don't use symmetric Dirichlet BCs can run in parallel
mpirun -np 3 python fenics-book/hodge.py N=4
#mpirun -np 3 python parallelmixedpoisson.py #fails, dolfin bug with numpy containers

ps -o etime,cputime $$
