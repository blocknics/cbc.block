from dolfin import *
from block import *
from block.iterative import *
from block.algebraic.petsc import ML
from numpy import random

dolfin.set_log_level(30)

N = 2
porder = 1
vorder = 2
alpha = 0

# Parse command-line arguments like "N=6", "alpha=0.1"
import sys
for s in sys.argv[1:]:
    exec(s)

class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class BoundaryFunction(Expression):
    def value_shape(self):
        return (2,)
    def eval(self, values, x):
        values[0] = 1 if near(x[1],1) else 0
        values[1] = 0

mesh = UnitSquareMesh(N,N)
def CG(n):
    return ('DG',0) if n==0 else ('CG',n)

V = VectorFunctionSpace(mesh, *CG(vorder))
Q = FunctionSpace(mesh, *CG(porder))

f = Constant((0,0))
g = Constant(0)
alpha = Constant(alpha)
h = CellSize(mesh)

v,u = TestFunction(V), TrialFunction(V)
q,p = TestFunction(Q), TrialFunction(Q)

a11 = inner(grad(v), grad(u))*dx
a12 = div(v)*p*dx
a21 = div(u)*q*dx
a22 = -alpha*h*h*dot(grad(p), grad(q))*dx
L1  = inner(v, f)*dx
L2  = q*g*dx

M1 = assemble(p*q*dx)

bcs = block_bc([DirichletBC(V, BoundaryFunction(degree=3), Boundary()), None], True)
AA = block_assemble([[a11, a12],
                     [a21, a22]])
bb = block_assemble([L1, L2])
bcs.apply(AA).apply(bb)

[[A, B],
 [C, D]] = AA

BB = block_mat([[ML(A),  0],
                [0, ML(M1)]])


AAinv = MinRes(AA, precond=BB, tolerance=1e-8, show=0)
x = AAinv * bb


x.randomize()

AAi = CGN(AA, precond=BB, initial_guess=x, tolerance=1e-8, maxiter=1000, show=0)
AAi * bb

e = AAi.eigenvalue_estimates()

print "N=%d iter=%d K=%.3g" % (N, AAinv.iterations, sqrt(e[-1]/e[0]))
