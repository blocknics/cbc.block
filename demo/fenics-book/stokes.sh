#!/bin/sh

#CG(2)-CG(1)
# N=16 iter=52 K=13.6
# N=32 iter=57 K=13.6
# N=64 iter=62 K=13.6
# N=128 iter=63 K=13.6
# N=256 iter=67 K=13.6
 
#CG(2)-DG(0)
# N=16 iter=43 K=8.55
# N=32 iter=48 K=9.14
# N=64 iter=55 K=9.71
# N=128 iter=58 K=10.3
# N=256 iter=60 K=10.7
 
#CG(1)-CG(1)
# N=16 iter=200+ K=696
# N=32 iter=200+ K=828
# N=64 iter=200+ K=673
# N=128 iter=200+ K=651
# N=256 iter=200+ K=630
 
#CG(1)-CG(1)-stab
# N=16 iter=41 K=12.5
# N=32 iter=40 K=12.6
# N=64 iter=40 K=12.7
# N=128 iter=39 K=12.7
# N=256 iter= K=

for param in "" "porder=0" "vorder=1" "vorder=1 alpha=0.01"; do
    echo "::" $param
    for N in 16 32; do
	python3 stokes.py $param N=$N
    done
done

