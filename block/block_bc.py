import dolfin
from .block_mat import block_mat
from .block_vec import block_vec
from .block_util import wrap_in_list
from .splitting import split_bcs
import itertools
import numpy

class block_bc:
    """This class applies Dirichlet BCs to a block matrix. It is not a block operator itself."""
    def __init__(self, bcs, symmetric=False, signs=None, subspace_bcs=None):
        # Clean up self, and check arguments
        self.symmetric = symmetric
        bcs = [wrap_in_list(bc) for bc in bcs]
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

    @classmethod
    def from_subspace(cls, bcs, *args, **kwargs):
        return cls([], *args, **kwargs, subspace_bcs=bcs)

    def apply(self, A):
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
        return block_rhs_bc(self.bcs, A_orig, symmetric=self.symmetric, signs=self.signs)

    def _apply(self, A):
        if self.symmetric:
            # dummy vec, required by dolfin api
            b = A.create_vec(dim=0)
        for i,bcs in enumerate(self.bcs):
            for bc in bcs:
                for j in range(len(A)):
                    if i==j:
                        if numpy.isscalar(A[i,i]):
                            # Convert to a diagonal matrix, so that the individual rows can be modified
                            from .block_assemble import _new_square_matrix
                            A[i,i] = _new_square_matrix(bc, A[i,i])
                        if self.symmetric:
                            bc.zero_columns(A[i,i], b[i], self.signs[i])
                        else:
                            bc.apply(A[i,i])
                    else:
                        if numpy.isscalar(A[i,j]):
                            if A[i,j] != 0.0:
                                dolfin.error("can't modify block (%d,%d) for BC, expected a GenericMatrix" % (i,j))
                        else:
                            bc.zero(A[i,j])
                        if self.symmetric:
                            if numpy.isscalar(A[j,i]):
                                if A[j,i] != 0.0:
                                    dolfin.error("can't modify block (%d,%d) for BC, expected a GenericMatrix" % (j,i))
                            else:
                                bc.zero_columns(A[j,i], b[j])

class block_rhs_bc:
    def __init__(self, bcs, A, symmetric, signs):
        self.bcs = bcs
        self.A = A
        self.symmetric = symmetric
        self.signs = signs

    def apply(self, b):
        """Apply Dirichlet boundary conditions, in a time loop for example,
        when boundary conditions change. If the original vector was modified
        for symmetry, it will remain so (since the BC dofs are not changed by
        symmetry), but if any vectors have been individually reassembled then
        it needs careful thought. It is probably better to just reassemble the
        whole block_vec using block_assemble()."""
        if not isinstance(b, block_vec):
            raise RuntimeError('not a block vector')

        b.allocate(self.A, dim=0)

        if self.symmetric:
            # First, collect a vector containing all non-zero BCs. These are required for
            # symmetric modification.
            b_mod = self.A.create_vec(dim=0)
            b_mod.zero()
            for i,bcs in enumerate(self.bcs):
                for bc in bcs:
                    bc.apply(b_mod[i])
                if self.signs[i] != 1:
                    b_mod[i] *= self.signs[i]

            # The non-zeroes of b_mod are now exactly the x values (scaled by
            # sign, i.e. matrix diagonal). We can thus create the necessary
            # modifications to b by just multiplying with the un-symmetricised
            # original matrix.
            b -= self.A * b_mod

        # Apply the actual BC dofs to b. (This must be done after the symmetric
        # correction above, since the correction might also change the BC dofs.)
        for i,bcs in enumerate(self.bcs):
            for bc in bcs:
                bc.apply(b[i])
            if self.signs[i] != 1:
                b[i] *= self.signs[i]

        return self
