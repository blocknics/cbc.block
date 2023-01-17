from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import str
from builtins import range
from .block_base import block_container

class block_vec(block_container):
    """Class defining a block vector suitable for multiplication with a
    block_mat of the right dimension. Many of the methods of dolfin.Vector are
    made available by calling the equivalent method on the individual
    vectors."""

    def __init__(self, m, blocks=None):
        if hasattr(m, '__iter__'):
            m = [x for x in m]
            blocks = m
            m = len(m)
        block_container.__init__(self, m, blocks)

    def allocated(self):
        """Check whether all blocks are proper vectors."""
        from dolfin import GenericVector
        return all(isinstance(block, GenericVector) for block in self)

    def allocate(self, template, dim=None):
        """Make sure all blocks are proper vectors. Any non-vector blocks are
        replaced with appropriately sized vectors (where the sizes are taken
        from the template, which should be a block_mat or a list of
        DirichletBCs or FunctionSpaces). If dim==0, newly allocated vectors use
        layout appropriate for b (in Ax=b); if dim==1, the layout for x is
        used."""
        if np.ndim(template) == 1 and dim is not None:
            raise ValueError('Cannot specify dim with 1D template')
        elif np.ndim(template) == 2 and dim is None:
            raise ValueError('2D template requires specified dim')
        from dolfin import GenericVector
        from .block_mat import block_mat
        from .block_util import create_vec_from
        for i in range(len(self)):
            if isinstance(self[i], GenericVector):
                continue
            val = self[i]
            try:
                self[i] = create_vec_from(template[:,i] if dim==1 else template[i], dim)
            except ValueError:
                pass
            if not isinstance(self[i], GenericVector):
                raise ValueError(
                    f"Can't allocate vector - no usable template for block {i}.\n"
                    "Consider calling something like bb.allocate([V, Q]) to initialise the block_vec."
                )
            self[i][:] = val or 0.0

    def norm(self, ntype='l2'):
        if ntype == 'linf':
            return max(x.norm(ntype) for x in self)
        else:
            try:
                assert(ntype[0] == 'l')
                p = int(ntype[1:])
            except:
                raise TypeError("Unknown norm '%s'"%ntype)
            unpack = lambda x: pow(x, p)
            pack   = lambda x: pow(x, 1/p)
            return pack(sum(unpack(x.norm(ntype)) for x in self))

    def randomize(self):
        """Fill the block_vec with random data (with zero bias)."""
        import numpy
        from dolfin import MPI
        # FIXME: deal with dolfin MPI api changes
        for i in range(len(self)):
            if hasattr(self[i], 'local_size'):
                ran = numpy.random.random(self[i].local_size())
                ran -= MPI.sum(self[i].mpi_comm(), sum(ran))/self[i].size()
                self[i].set_local(ran)
            elif hasattr(self[i], '__len__'):
                ran = numpy.random.random(len(self[i]))
                ran -= MPI.sum(self[i].mpi_comm(), sum(ran))/MPI.sum(len(ran))
                self[i][:] = ran
            else:
                raise ValueError(
                    f'block {i} in block_vec has no size -- use a proper vector or call allocate(A, dim=d)')

        for i in range(len(self)):
            if not isinstance(self[i], GenericVector):
                vec = create_vec_from(bcs[i])
                vec[:] = self[i]
                self[i] = vec
            for bc in wrap_in_list(bcs[i]):
                bc.apply(self[i])

    #
    # Map operator on the block_vec to operators on the individual blocks.
    #

    def _map_operator(self, operator, inplace=False):
        y = block_vec(len(self))
        for i in range(len(self)):
            try:
                y[i] = getattr(self[i], operator)()
            except (Exception, e):
                if i==0 or not inplace:
                    raise e
                else:
                    raise RuntimeError(
                        "operator partially applied, block %d does not support '%s' (err=%s)" % (i, operator, str(e)))
        return y

    def _map_scalar_operator(self, operator, x, inplace=False):
        try:
            x = float(x)
        except:
            return NotImplemented
        y = self if inplace else block_vec(len(self))
        for i in range(len(self)):
            v = getattr(self[i], operator)(x)
            if isinstance(v, type(NotImplemented)):
                if i==0 or not inplace:
                    return NotImplemented
                else:
                    raise RuntimeError(
                        "operator partially applied, block %d does not support '%s'" % (i, operator))
            y[i] = v
        return y

    def _map_vector_operator(self, operator, x, inplace=False):
        y = self if inplace else block_vec(len(self))
        for i in range(len(self)):
            v = getattr(self[i], operator)(x[i])
            if isinstance(v, type(NotImplemented)):
                if i==0 or not inplace:
                    return NotImplemented
                else:
                    raise RuntimeError(
                        "operator partially applied, block %d does not support '%s'" % (i, operator))
            y[i] = v
        return y


    def copy(self):
        from . import block_util
        m = len(self)
        y = block_vec(m)
        for i in range(m):
            y[i] = block_util.copy(self[i])
        return y
    
    def zero(self): return self._map_operator('zero', True)

    def __add__ (self, x): return self._map_vector_operator('__add__',  x)
    def __radd__(self, x): return self._map_vector_operator('__radd__', x)
    def __iadd__(self, x): return self._map_vector_operator('__iadd__', x, True)

    def __sub__ (self, x): return self._map_vector_operator('__sub__',  x)
    def __rsub__(self, x): return self._map_vector_operator('__rsub__', x)
    def __isub__(self, x): return self._map_vector_operator('__isub__', x, True)

    def __mul__ (self, x): return self._map_scalar_operator('__mul__',  x)
    def __rmul__(self, x): return self._map_scalar_operator('__rmul__', x)
    def __imul__(self, x): return self._map_scalar_operator('__imul__', x, True)

    def inner(self, x):
        y = self._map_vector_operator('inner', x)
        if isinstance(y, type(NotImplemented)):
            raise NotImplementedError('One or more blocks do not implement .inner()')
        return sum(y)
