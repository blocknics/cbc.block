import itertools
import pytest
from inspect import signature
from block.algebraic.petsc import *

@pytest.mark.parametrize('prec_class',
                         itertools.chain(precond.__subclasses__(), LU.__subclasses__()))
def test_construct(prec_class, poisson):
    if prec_class in [ML, HypreAMS, HypreADS]:
        # ML requires Trilinos
        # HypreAMS requires RT elements; tested in mixedpoisson_hypre_2d.py
        # HypreADS requires 3D; not tested
        pytest.skip('special needs')

    prec_args = signature(prec_class).parameters
    if 'V' in prec_args:
        Apre = prec_class(poisson.A, V=poisson.V)
    else:
        Apre = prec_class(poisson.A)

    x = Apre * poisson.b
    assert (x - poisson.x).norm('l2') / poisson.x.norm('l2') < 1
