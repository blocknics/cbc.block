from block import *
from block.iterative import *
from block.algebraic.hazmath import AMG as hazAMG
from dolfin import *
from block.dolfin_util import *
import numpy

# Function spaces, elements

mesh = UnitSquareMesh(16,16)

V = FunctionSpace(mesh, "CG", 1)

f = Expression("sin(pi*x[0])", degree=2)
u, v = TrialFunction(V), TestFunction(V)

a = u*v*dx + dot(grad(u), grad(v))*dx
L = f*v*dx

A = assemble(a)
b = assemble(L)

# to do hypre AMG
# B = AMG(A)
# here we use hazmath AMG:
B = hazAMG(A)

Ainv = ConjGrad(A, precond=B, tolerance=1e-10, show=2)

x = Ainv*b

u = Function(V)
u.vector()[:] = x[:]

# default solver in Dolfin
u2 = Function(V)
solve(A, u2.vector(), b)

check_expected('x', x, expected=u2.vector(), show=True)


