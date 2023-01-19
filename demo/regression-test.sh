#!/bin/bash

# Script that runs all cbc.block demos and exits if anything goes wrong. If
# this script finishes successfully, we are reasonably confident that there are
# no api-related breakages, and that the results run through 'check_expected' are
# not regressed.

set -e
export DOLFIN_NOPLOT=1
export BLOCK_REGRESSION_ABORT=1

cd ${0%/*}
demos=$(find . -name \*.py)

xargs -P4 -n1 sh -c 'echo $0; python3 $0 >/dev/null || exit 255' <<<$demos

#for demo in $demos; do
#    echo mpirun -np 3 python $demo
#    mpirun -np 3 python $demo
#done

ps -o etime,cputime $$
