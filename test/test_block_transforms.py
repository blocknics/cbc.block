from block import *
import numpy as np
import pytest

eps = np.finfo(np.float64).eps

@pytest.fixture
def blocks2x2(poisson):
    A, b = poisson.A, poisson.b
    AA = block_mat([[2*A, 0],
                    [A,   3]])
    bb = block_vec([0, b])
    bb.allocate(AA, dim=1)
    return AA, bb

def test_basic_block_ops(blocks2x2):
    AA,bb = blocks2x2
    [A,B],[C,D] = AA
    b,c = bb
    assert (AA*bb - [A*b+B*c, C*b+D*c]).norm('linf') < eps
    assert ((AA+AA.T)*bb - [2*A*b+(B+C)*c, (B+C)*b+2*D*c]).norm('linf') < eps

def test_block_collapse(blocks2x2):
    AA,bb = blocks2x2

    AA = (AA*AA-AA.T)+AA
    AAc = block_collapse(AA)
    assert isinstance(AA, block_add)
    assert isinstance(AAc, block_mat)
    assert isinstance(AAc[0,0], block_add)
    assert block_simplify(AAc[1,1]) == 9 # 3*3+3-3; AA is lower triangular
    assert (AA*bb - AAc*bb).norm('linf') < eps
