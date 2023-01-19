from block import *
from block.iterative import *
from block.algebraic.petsc import AMG as hypreAMG
from block.algebraic.hazmath import AMG as hazAMG
from dolfin import *
from block.dolfin_util import *
import numpy as np

# Function spaces, elements

mesh = UnitSquareMesh(32,32)

V = FunctionSpace(mesh, "CG", 1)

f = Expression("sin(pi*x[0])", degree=2)
u, v = TrialFunction(V), TestFunction(V)

a = u*v*dx + dot(grad(u), grad(v))*dx
L = f*v*dx

A = assemble(a)
b = assemble(L)

# to do hypre AMG
#B = hypreAMG(A)
# here we use hazmath AMG:
B = hazAMG(A)

Ainv = ConjGrad(A, precond=B, tolerance=1e-13, show=0)

x = Ainv*b

check_expected('x', x)

u = Function(V)
u.vector()[:] = x[:]

# default solver in Dolfin
u2 = Function(V)
solve(A, u2.vector(), b)

print ("Max differences between the two solutions: ", (u.vector()-u2.vector()).norm("linf"))


