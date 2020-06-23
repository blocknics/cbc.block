#!/bin/sh

# N=16 iter=8 K=1.57
# N=32 iter=8 K=1.26
# N=64 iter=10 K=2.09
# N=128 iter=9 K=1.49
# N=256 iter=7 K=1.19

for N in 16 32 64 128 256; do
    python3 poisson_neumann.py N=$N
done
