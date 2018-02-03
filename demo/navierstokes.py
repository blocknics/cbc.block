"""This demo solves the Navier-Stokes equations with block preconditioning.

NOTE: Needs work. Must define a better precond, this one doesn't work for smaller mu.

It is a modification of the Stokes demo (demo-stokes.py).
"""

# Scipy (in LGMRES) seems to crash unless it's loaded before PyTrilinos.
import scipy
import os

from dolfin import *
from block import *
from block.iterative import *
from block.algebraic.petsc import *

dolfin.set_log_level(15)
if MPI.size(None) > 1:
    print ("Navier-Stokes demo does not work in parallel because of old-style XML mesh files")
    exit()

# Load mesh and subdomains
path = os.path.join(os.path.dirname(__file__), '')
mesh = Mesh(path+"dolfin-2.xml.gz")
sub_domains = MeshFunction("size_t", mesh, path+"subdomains.xml.gz")

# Define function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)

# No-slip boundary condition for velocity
noslip = Constant((0, 0))
bc0 = DirichletBC(V, noslip, sub_domains, 0)

# Inflow boundary condition for velocity
inflow = Expression(("-sin(x[1]*pi)", "0.0"))
bc1 = DirichletBC(V, inflow, sub_domains, 1)

# Boundary condition for pressure at outflow
zero = Constant(0)
bc2 = DirichletBC(Q, zero, sub_domains, 2)

# Define variational problem and assemble matrices
v, u = TestFunction(V), TrialFunction(V)
q, p = TestFunction(Q), TrialFunction(Q)

f = Constant((0, 0))
w = Constant((1, 0))
mu = Constant(1e-2)

a11 = mu*inner(grad(v), grad(u))*dx + inner(dot(grad(u),w),v)*dx
a12 = div(v)*p*dx
a21 = div(u)*q*dx
L1  = inner(v, f)*dx

# Create the block matrix/vector, with boundary conditions applied. The zeroes
# in A[1,1] and b[1] are automatically converted to matrices/vectors to be able
# to apply bc2.
bcs = [[bc0, bc1], bc2]
AA = block_assemble([[a11, a12],
                     [a21,  0 ]], bcs=bcs)
bb = block_assemble([L1, 0], bcs=bcs)

# Extract the individual submatrices
[[A, B],
 [C, _]] = AA

# Create preconditioners: An ILU preconditioner for A, and an ML inverse of the
# Schur complement approximation for the (2,2) block.
Ap = ILU(A)
Dp = ML(collapse(C*InvDiag(A)*B))

prec = block_mat([[Ap, B],
                  [C, -Dp]]).scheme('sgs')

# Create the block inverse, using the LGMRES method (suitable for general problems).
AAinv = LGMRES(AA, precond=prec, tolerance=1e-5, maxiter=50, show=2)

# Compute solution
u, p = AAinv * bb

print "Norm of velocity coefficient vector: %.15g" % u.norm("l2")
print "Norm of pressure coefficient vector: %.15g" % p.norm("l2")

# Plot solution
plot(Function(V, u))
plot(Function(Q, p))
interactive()
