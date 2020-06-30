#!/bin/sh

#k=1
# N=16 K=13.6
# N=32 K=13.6
# N=64 K=13.6
# N=128 K=13.7
# N=256 K=13.7

#k=0.1
# N=16 K=
# N=32 K=
# N=64 K=
# N=128 K=
# N=256 K=

#k=0.01
# N=16 K=
# N=32 K=
# N=64 K=
# N=128 K=
# N=256 K=

#k=0.001
# N=16 K=
# N=32 K=
# N=64 K=
# N=128 K=
# N=256 K=


for k in 1 0.1 0.01 0.001; do
    echo ":: k_val=" $k
    for N in 16 32 64 128 256; do
	python3 timestokes.py k_val=$k N=$N
    done
done
