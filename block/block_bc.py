import dolfin
from .block_mat import block_mat
from .block_vec import block_vec
from .block_util import wrap_in_list, create_diagonal_matrix
from .splitting import split_bcs
import itertools
import numpy

class block_bc:
    """
    This class applies Dirichlet BCs to a block matrix.  It is not a block
    operator itself.

    Creating bcs:

    >>> bcs = block_bc([...], symmetric=...)

    Basic inplace usage, with static BCs:

    >>> rhs_bcs = bcs.apply(A)
    >>> rhs_bcs.apply(b)

    with the shortcut

    >>> bcs.apply(A,b)

    If boundary conditions are applied multiple times, it may be useful
    to keep the original assembled mat/vec unchanged (for example, to
    avoid having to re-assemble the RHS if the BC is time dependent but
    the form itself isn't):

    >>> Ainv = conjgrad(bcs(A))
    >>> while True:
    >>>     source.t = t
    >>>     x = Ainv * bcs(b)
    """
    def __init__(self, bcs, symmetric=False, signs=None, subspace_bcs=None):
        # Clean up self, and check arguments
        self.symmetric = symmetric
        bcs = [wrap_in_list(bc, dolfin.DirichletBC) for bc in bcs]
        if subspace_bcs is not None:
            subspace_bcs = split_bcs(subspace_bcs, None)
            combined_bcs = []
            for ss,ns in itertools.zip_longest(subspace_bcs, bcs):
                combined_bcs.append([])
                if ns is not None: combined_bcs[-1] += ns
                if ss is not None: combined_bcs[-1] += ss
            self.bcs = combined_bcs
        else:
            self.bcs = bcs
        self.signs = signs or [1]*len(self.bcs)
        if not all(s in [-1,1] for s in self.signs):
            raise ValueError('signs should be a list of length n containing only 1 or -1')

    @classmethod
    def from_mixed(cls, bcs, *args, **kwargs):
        return cls([], *args, **kwargs, subspace_bcs=bcs)

    def __call__(self, other, A=None):
        if isinstance(other, block_mat):
            self._A = other # opportunistic, expecting a block_vec later
            A = other.copy()
            self.apply(A)
            return A
        elif isinstance(other, block_vec):
            if A is None:
                A = getattr(self, '_A', None) # opportunistic
            if self.symmetric and A is None:
                raise ValueError('A not available. Call with A first, or pass A=A.')
            rhs = self.rhs(A)
            return rhs(other)
        else:
            raise TypeError('BCs can only act on block_mat (A) or block_vec (b)')

    def apply(self, A, b=None):
        """
        Apply BCs to a block_mat LHS, and return an object which modifies the
        corresponding block_vec RHS.  Typical use:

        b_bcs = A_bcs.apply(A); b_bcs.apply(b)
        """
        if not isinstance(A, block_mat):
            raise RuntimeError('A is not a block matrix')

        A_orig = A.copy() if self.symmetric else A
        self._apply(A)

        # Create rhs_bc with a copy of A before any symmetric modifications
        rhs = self.rhs(A_orig)
        if b is not None:
            rhs.apply(b)
        return rhs

    def rhs(self, A):
        return block_rhs_bc(self.bcs, A, symmetric=self.symmetric, signs=self.signs)

    def _apply(self, A):
        if self.symmetric:
            # dummy vec, required by dolfin api -- we don't use this
            # corrections directly since it may be time dependent
            b = A.create_vec(dim=0)
        for i,bcs in enumerate(self.bcs):
            if bcs:
                for j in range(len(A)):
                    if i==j:
                        if numpy.isscalar(A[i,i]):
                            # Convert to a diagonal matrix, so that the individual rows can be modified
                            A[i,i] = create_diagonal_matrix(dolfin.FunctionSpace(bc.function_space()), A[i,i])
                        if self.symmetric:
                            for bc in bcs:
                                bc.zero_columns(A[i,i], b[i], self.signs[i])
                        else:
                            if self.signs[i] != 1:
                                A[i,i] *= 1/self.signs[i]
                            for bc in bcs:
                                bc.apply(A[i,i])
                            if self.signs[i] != 1:
                                A[i,i] *= self.signs[i]
                    else:
                        if numpy.isscalar(A[i,j]):
                            if A[i,j] != 0.0:
                                dolfin.error("can't modify nonzero scalar off-diagonal block (%d,%d)" % (i,j))
                        else:
                            for bc in bcs:
                                bc.zero(A[i,j])
                        if self.symmetric:
                            if numpy.isscalar(A[j,i]):
                                if A[j,i] != 0.0:
                                    dolfin.error("can't modify nonzero scalar off-diagonal block (%d,%d)" % (i,j))
                            else:
                                for bc in bcs:
                                    bc.zero_columns(A[j,i], b[j])

class block_rhs_bc:
    """
    This class applies Dirichlet BCs to a block block vector.  It can be used
    as a block operator; but since it is nonlinear, it can only operate
    directly on a block vector, and not combine with other operators.
    """
    def __init__(self, bcs, A, symmetric, signs):
        if symmetric:
            assert A is not None
        self.bcs = bcs
        self.A = A
        self.symmetric = symmetric
        self.signs = signs

    @property
    def b_mod(self):
        # First, collect a vector containing all non-zero BCs. These are required for
        # symmetric modification.
        x_mod = self.A.create_vec(dim=0)
        x_mod.zero()
        for i,bcs in enumerate(self.bcs):
            for bc in bcs:
                bc.apply(x_mod[i])

        # The non-zeroes of x_mod are now exactly the x values. We can thus
        # create the necessary modifications to b by just multiplying with the
        # un-symmetricised original matrix. The bc values are overwritten
        # later, hence only the non-bc rows of A matter.
        return self.A * x_mod

    def apply(self, b):
        """Apply Dirichlet boundary conditions statically.  If BCs are mutable
        (time dependent for example), it is more convenient to use the callable
        form -- rhs_bc(b) -- to preserve the original contents of b for repeated
        application without reassembly.
        """
        if not isinstance(b, block_vec):
            raise TypeError('not a block vector')

        try:
            b.allocate(self.A, dim=0, alternative_templates=[self.bcs])
        except Exception:
            raise ValueError('Failed to allocate block vector, call b.allocate(something) first')

        # Correct for the matrix elements zeroed by symmetricization
        if self.symmetric:
            b -= self.b_mod

        # Apply the actual BC dofs to b. (This must be done after the symmetric
        # correction above, since the correction might also change the BC dofs.)
        # If the sign is negative, we negate twice to effectively apply -bc.
        for i,bcs in enumerate(self.bcs):
            if self.signs[i] != 1 and bcs:
                b[i] *= 1/self.signs[i]
            for bc in bcs:
                bc.apply(b[i])
            if self.signs[i] != 1 and bcs:
                b[i] *= self.signs[i]

        return self

    def __call__(self, other):
        if not isinstance(other, block_vec):
            raise TypeError()
        b = other.copy()
        self.apply(b)
        return b
