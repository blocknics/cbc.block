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

       | M   0 |
  BB = |       |,
       | 0   N |

M is the Riesz map with respect to H(div) inner product and N is the 
Riesz map with respect to the L^2 inner product
"""
from __future__ import division
from __future__ import print_function

from block import *
from block.iterative import *
from block.algebraic.petsc import AMG, HypreADS
from dolfin import *

n = 16
# Create mesh
mesh = UnitCubeMesh(n, n, n)

# Define function spaces
RT = FunctionSpace(mesh, "RT", 1)
DG = FunctionSpace(mesh, "DG", 0)

# Define trial and test functions
tau, sigma = TestFunction(RT), TrialFunction(RT)
v,   u     = TestFunction(DG),  TrialFunction(DG)

# Define function G such that G \cdot n = g
class BoundarySource(UserExpression):
    def __init__(self, mesh, **kwargs):
        super().__init__(self)
        self.mesh = mesh
    def eval_cell(self, values, x, ufc_cell):
        cell = Cell(self.mesh, ufc_cell.index)
        n = cell.normal(ufc_cell.local_facet)
        g = sin(5*x[0])
        values[0] = g*n[0]
        values[1] = g*n[1]
        values[2] = g*n[2]        
    def value_shape(self):
        return (3,)

G = BoundarySource(mesh, degree=1)

# Define essential boundary
def boundary(x):
    return near(x[1], 0.0) 

# Define the blockwise boundary conditions -- a Dirichlet condition on the
# first block, and no conditions on the second block.
bcs_RT = [DirichletBC(RT, G, boundary)]
bcs = block_bc([bcs_RT, None], True)

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

# Unlike in mixedpoisson.py the blocks of the preconditioner are constructed
# rather than extracted from the system
prec11 = inner(sigma, tau)*dx + inner(div(sigma), div(tau))*dx
prec22 = inner(u, v)*dx

BB = block_assemble([[prec11, 0],
                     [0, prec22]])
bcs.apply(BB)

# We invert the blocks by taylored multigrid
M = HypreADS(A=BB[0][0], V=RT)
N = AMG(BB[1][1])

# Define the block preconditioner
AAp = block_mat([[M, 0],
                 [0,  N]])

AAinv = MinRes(AA, precond=AAp, show=2, name='AA^', tolerance=1E-10)

#=====================
# Solve the system
Sigma, U = AAinv * bb
#=====================

# Print norms that can be compared with those reported by demo-parallelmixedpoisson
print(('norm Sigma:', Sigma.norm('l2')))
print(('norm U    :', U.norm('l2')))

# Plot sigma and u
if MPI.size(mesh.mpi_comm()) == 1:
    File('sigma_h.pvd') << Function(RT, Sigma)
    File('u_h.pvd') << Function(DG,  U)
