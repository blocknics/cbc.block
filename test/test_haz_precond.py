from dolfin import *
from block.iterative import ConjGrad
from block.algebraic.hazmath import AMG, RA, HXDiv, HXCurl, FAMG
import haznics

mesh = UnitSquareMesh(16, 16)

V = FunctionSpace(mesh, "CG", 1)

u, v = TrialFunction(V), TestFunction(V)
a = inner(u, v)*dx + inner(grad(u), grad(v))*dx

A = assemble(a)

# ********* AMG ********* #
B = AMG(A)
B1 = AMG(A, parameters={'cycle_type': haznics.W_CYCLE,
                        'maxit': 2,
                        'smoother': haznics.SMOOTHER_GS,
                        })
# check properties
print("************* AMG DEFAULT PARAMS **************** ")
print("Create vector: ")
bb = B.create_vec(dim=1)
print("Done.\n")

print("Apply precond once: ")
xx = B.matvec(bb)
print("Done.\n")

print("Print AMG parameters: ")
B.print_amg_parameters()
print("Done. \n")

try:
    B.__amg_parameters
except AttributeError:
    print("Hidden variable")

print("************* AMG MODIFIED PARAMS **************** ")
print("Create vector: ")
bb = B1.create_vec(dim=1)
print("Done.\n")

print("Apply precond once: ")
xx = B1.matvec(bb)
print("Done.\n")

print("Print AMG parameters: ")
B1.print_amg_parameters()
print("Done. \n")


# check solve
print("************* AMG SOLVE **************** ")
f = Expression("sin(pi*x[0])", degree=5)
L = f*v*dx
b = assemble(L)

Ainv = ConjGrad(A, precond=B, tolerance=1e-10, show=2)

x = Ainv*b

u = Function(V)
u.vector()[:] = x[:]

# default solver in Dolfin
u2 = Function(V)
solve(A, u2.vector(), b)

print("Max differences between the two solutions: ", (u.vector()-u2.vector()).max())


# ********* RA ********* #
u, v = TrialFunction(V), TestFunction(V)
m = inner(u, v)*dx
M = assemble(m)

B2 = RA(A, M, parameters={'coefs': [1.0, 2.0], 'pwrs': [-0.5, 0.5]})

print("************* RA PARAMS **************** ")
print("Create vector: ")
bbb = B2.create_vec(dim=1)
print("Done.\n")

print("Apply precond once: ")
xxx = B.matvec(bbb)
print("Done.\n")

print("Print RA parameters: ")
B2.print_amg_parameters()
B2.print_all_parameters()
print("Done. \n")



