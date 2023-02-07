#!/bin/bash

# Script that runs all cbc.block demos and exits if anything goes wrong. If
# this script finishes successfully, we are reasonably confident that there are
# no api-related breakages, and that the results run through 'check_expected' are
# not regressed.

# Usage: demo/regression-test.sh [--coverage] [--mpirun]
#
# If --coverage is set, the demos that contain "check_expected" calls have
# their coverage checked. Use "coverage combine" then "coverage report" to
# inspect results.
#
# If --mpirun is set, the demos are run with mpi. Most will fail.

set -e

PARALLEL=-P4
if [[ $1 == "--coverage" ]]; then
    export BLOCK_REGRESSION_COVERAGE=1
    shift
elif [[ $1 == "--mpirun" ]]; then
    export BLOCK_REGRESSION_MPIRUN=1
    PARALLEL=
    shift
fi

if (( $# )); then
    if (( $# != 1 )); then
        echo "Expected a single argument (the script)"
        exit 255
    fi
    if [[ -n $BLOCK_REGRESSION_COVERAGE ]] && grep -q '^[^#]*check_expected(' "$1"; then
        cmd="coverage run --parallel-mode --branch"
    else
        cmd="python3"
    fi
    if [[ -n $BLOCK_REGRESSION_MPIRUN ]]; then
        cmd="mpirun -np 3 $cmd"
    fi
    echo $cmd $1
    # exit code 255 makes xargs abort immediately
    $cmd "$1" >/dev/null || exit 255
    exit 0
fi

export DOLFIN_NOPLOT=1
export BLOCK_REGRESSION_ABORT=1

demos=$(find "${0%/*}" -name \*.py)

xargs $PARALLEL -n1 "$0" <<<$demos

ps -o etime,cputime $$
