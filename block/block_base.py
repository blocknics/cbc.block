from __future__ import division

import numpy

class block_base(object):
    """Base class for (block) operators. Defines algebraic operations that
    defer actual calculations to a later time if the RHS is not a
    vector. Classes that inherit from block_base should at least define a
    matvec(self, other) method.
    """
    def __mul__(self, other):
        from block_compose import block_mul
        from block_vec import block_vec
        from dolfin import GenericVector
        if not isinstance(other, (block_vec, GenericVector)):
            return block_mul(self, other)
        return self.matvec(other)

    def __rmul__(self, other):
        from block_compose import block_mul
        return block_mul(other, self)

    def __neg__(self):
        from block_compose import block_mul
        return block_mul(-1, self)

    def __add__(self, other):
        from block_compose import block_add
        return block_add(self, other)

    def __radd__(self, other):
        from block_compose import block_add
        return block_add(other, self)

    def __sub__(self, other):
        from block_compose import block_sub
        return block_sub(self, other)

    def __rsub__(self, other):
        from block_compose import block_sub
        return block_sub(other, self)

    def __pow__(self, other):
        p = int(other)
        if p != other or p < 0:
            raise ValueError("power must be a positive integer")
        if p == 0:
            return 1
        if p == 1:
            return self
        return self * pow(self, other-1)

class block_container(block_base):
    """Base class for block containers: block_mat and block_vec.
    """
    def __init__(self, mn, blocks):
        self.blocks = numpy.ndarray(mn, dtype=numpy.object)
        self.blocks[:] = blocks

    def __setitem__(self, key, val):
        self.blocks[key] = val
    def __getitem__(self, key):
        try:
            return self.blocks[key]
        except IndexError, e:
            raise IndexError(str(e) + ' at ' + str(key) + ' -- incompatible block structure')
    def __len__(self):
        return len(self.blocks)
    def __iter__(self):
        return self.blocks.__iter__()
    def __str__(self):
        return '<%s %s:\n%s>'%(self.__class__.__name__,
                               'x'.join(map(str, self.blocks.shape)),
                               str(self.blocks))
