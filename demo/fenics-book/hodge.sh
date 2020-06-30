#!/bin/sh
for N in 2 4 8 16 32; do
    python3 hodge.py N=$N
done
