from block import *
from block.iterative import *
from block.algebraic.petsc import *
from dolfin import *
from block.dolfin_util import *
import numpy

# Function spaces, elements

mesh = UnitSquareMesh(16,16)

V = FunctionSpace(mesh, "CG", 1)

f = Expression("sin(3.14*x[0])", degree=2)
u, v = TrialFunction(V), TestFunction(V)

a = u*v*dx + dot(grad(u), grad(v))*dx
L = f*v*dx

A = assemble(a)
b = assemble(L)

B = AMG(A)

Ainv = ConjGrad(A, precond=B, tolerance=1e-10, show=2)

x = Ainv*b

u = Function(V)
u.vector()[:] = x[:]
plot(u, title="u, computed by cbc.block [x=Ainv*b]")

u2 = Function(V)
solve(A, u2.vector(), b)
plot(u2, title="u2, computed by dolfin [solve(A,x,b)]")


