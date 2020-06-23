"""This demo program shows the use of block preconditioners for Mixed
Poisson. The original DOLFIN demo, with description of the mixed formulation of
the variational problem, can be found in $DOLFIN_DIR/demo/pde/mixed-poisson/python.

The algebraic system to be solved can be written as

  BB^ AA [sigma u]^T = BB^ [0 b]^T,

where AA is a 2x2 block system with zero in the (2,2) block

       | A   B |
  AA = |       |,
       | C   0 |

and BB^ approximates the inverse of the block operator

       | A   0 |
  BB = |       |,
       | 0   L |

where L is the Laplace operator. Since the DG(0) approximation of L is zero, we
calculate it instead as L = C*B.

When forming the preconditioner BB^, we require approximate inverses of A and
L. For A, this is straightforward: an ML multilevel preconditioner is used:

  A^ = ML(A)

For L, however, we never actually calculate the matrix product, we just create
a composite operator so that

  x = L*v
            ==> w = B*v; x = C*w

Hence, all preconditioners that require access to the matrix elements (which is
most of them) are unavailable. Instead we use an inner iterative solver:

  L^ = Richardson(L, precond=0.5, iter=40)

This describes a solver using Richardson iterations, with damping 0.5 and a
fixed number of iterations. It is not very efficient, but since it is a linear
operator it is safe to use as inner solver for an outer Krylov solver.
"""
from __future__ import division
from __future__ import print_function

from block import *
from block.iterative import *
from block.algebraic.petsc import AMG, collapse
from dolfin import *

# Create mesh
mesh = UnitSquareMesh(32,32)

# Define function spaces
BDM = FunctionSpace(mesh, "BDM", 1)
DG = FunctionSpace(mesh, "DG", 0)

# Define trial and test functions
tau, sigma = TestFunction(BDM), TrialFunction(BDM)
v,   u     = TestFunction(DG),  TrialFunction(DG)

# Define function G such that G \cdot n = g
class BoundarySource(UserExpression):
    def __init__(self, mesh):
        super().__init__(self)
        self.mesh = mesh
    def eval_cell(self, values, x, ufc_cell):
        cell = Cell(self.mesh, ufc_cell.index)
        n = cell.normal(ufc_cell.local_facet)
        g = sin(5*x[0])
        values[0] = g*n[0]
        values[1] = g*n[1]
    def value_shape(self):
        return (2,)

G = BoundarySource(mesh)

# Define essential boundary
def boundary(x):
    return near(x[1], 0.0) or near(x[1], 1.0)

# Define the blockwise boundary conditions -- a Dirichlet condition on the
# first block, and no conditions on the second block.
bcs_BDM = [DirichletBC(BDM, G, boundary)]
bcs = block_bc([bcs_BDM, None], True)

# Define source function
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=4)

# Define variational forms (one per block)
a11 = dot(sigma, tau) * dx
a12 = div(tau) * u * dx
a21 = div(sigma) * v *dx
L2  = - f * v * dx

AA = block_assemble([[a11, a12],
                     [a21,  0 ]])

bb = block_assemble([0, L2])

bcs.apply(AA).apply(bb)

# Extract the individual submatrices
[[A, B],
 [C, _]] = AA

# Create a preconditioner for A (using the ML preconditioner from Trilinos)
Ap = AMG(A)

# Create an approximate inverse of L=C*B using inner Richardson iterations
L = C*B
Lp = Richardson(L, precond=0.5, iter=40, name='L^')

# Alternative b: Use inner Conjugate Gradient iterations. Not completely safe,
# but faster (and does not require damping).
#
#Lp = ConjGrad(L, maxiter=40, name='L^')

# Alternative c: Calculate the matrix product, so that a regular preconditioner
# can be used. The "collapse" function collapses a composed operator into a
# single matrix. For larger problems, and in particular on parallel computers,
# this is a very expensive operation --- but here it works fine.
#
#Lp = ML(collapse(L))

# Define the block preconditioner
AAp = block_mat([[Ap, 0],
                 [0,  Lp]])

# Define the block inverse using an outer Preconditioned Minimum Residual
# method, suitable for symmetric indefinite problems. SymmLQ is a good
# alternative (often slower, but more robust for ill-conditioned problems).
# ConjGrad may be much faster and will often work fine, but it is not
# guaranteed to converge since the matrix is not definite.
AAinv = MinRes(AA, precond=AAp, show=2, name='AA^')

#=====================
# Solve the system
Sigma, U = AAinv * bb
#=====================

# Print norms that can be compared with those reported by demo-parallelmixedpoisson
print(('norm Sigma:', Sigma.norm('l2')))
print(('norm U    :', U.norm('l2')))

# Check that the norms are as expected
if abs(1.213-Sigma.norm('l2')) > 1e-3 or abs(6.716-U.norm('l2')) > 1e-3:
    raise RuntimeError("Wrong value in norms -- please check!")

# Plot sigma and u
if MPI.size(mesh.mpi_comm()) == 1:
    plot(Function(BDM, Sigma))
    plot(Function(DG,  U))
