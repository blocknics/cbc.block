from __future__ import division
from __future__ import print_function
from dolfin import *
from block import check_expected
from block.iterative import ConjGrad
from block.algebraic.petsc import ILU

# Source term
class Source(UserExpression):
    def __init__(self):
        super().__init__(self)    
    def eval(self, values, x):
        dx = x[0] - 0.5
        dy = x[1] - 0.5
        values[0] = 500.0*exp(-(dx*dx + dy*dy)/0.02)
    def value_shape(self): return ()

# Neumann boundary condition
class Flux(UserExpression):
    def __init__(self):
        super().__init__(self)
    def eval(self, values, x):
        if x[0] > DOLFIN_EPS:
            values[0] = 25.0*sin(5.0*DOLFIN_PI*x[1])
        else:
            values[0] = 0.0
    def value_shape(self): return ()
    
N = 16

# Parse command-line arguments like "N=6"
import sys
for s in sys.argv[1:]:
    exec(s)

# Create mesh and finite element
mesh = UnitSquareMesh(N,N)
V = FunctionSpace(mesh, "CG", 1)

# Define variational problem
v = TestFunction(V)
u = TrialFunction(V)
f = Source()
g = Flux()

a = dot(grad(v), grad(u))*dx
L = v*f*dx + v*g*ds

# Assemble matrix and vector
A, b = assemble_system(a,L)

# remove constant from right handside
b_mean = MPI.sum(MPI.comm_world, sum(b.get_local()))/b.size()
c = A.create_vec()
c[:] = b_mean
b -= c

# create preconditioner
# Note jan-2023: PETSc-AMG fails -- AMG(A) not positive definite.
# See https://lists.mcs.anl.gov/mailman/htdig/petsc-users/2014-October/023318.html
B = ILU(A)
Ainv = ConjGrad(A, precond=B, tolerance=1e-8, show=0)

check_expected('x', Ainv*b)

e = Ainv.eigenvalue_estimates()

print ("N=%d iter=%d K=%.3g" % (N, Ainv.iterations, e[-1]/e[0]))
