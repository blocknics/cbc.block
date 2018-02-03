"""Demo showing how to mix dolfin matrices with dense numpy arrays in a block_mat"""

from block import *
from block.iterative import *
from block.algebraic.petsc import *
from dolfin import *
from block.dolfin_util import *
import numpy

if MPI.size(None) > 1:
    print ("numpy demo does not work in parallel")
    exit()

# Function spaces, elements

mesh = UnitSquareMesh(16,16)

V = FunctionSpace(mesh, "CG", 1)

f = Expression("sin(3.14*x[0])")
u, v = TrialFunction(V), TestFunction(V)

a = u*v*dx
L = f*v*dx

A = assemble(a)
b = assemble(L)
C = numpy.zeros([A.size(0), A.size(1)])

AA = block_mat([[A, A],
                [A, C]])
bb = block_vec([b, b])

xx = AA*bb
assert abs(xx.norm() - 3.29e-4) < 1e-6

