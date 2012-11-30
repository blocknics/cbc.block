from __future__ import division
from dolfin import warning

def isscalar(obj):
    """Return True if obj is convertible to float. Use this instead of
    numpy.isscalar, becuase the latter returns true for e.g. strings"""
    try:
        float(obj)
        return True
    except:
        return False

def mult(op, x, transposed=False):
    if not transposed or isscalar(op):
        return op*x
    else:
        return op.transpmult(x)

def copy(obj):
    if hasattr(obj, 'copy'):
        return obj.copy()
    else:
        import copy
        try:
            return copy.deepcopy(obj)
        except TypeError:
            warning("Don't know how to make a deep copy of (%d,%d), making shallow copy"%(i,j))
            return copy.copy(obj)
