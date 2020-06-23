"""This demo shows the use of a non-trivial block preconditioner for the time
dependent Stokes equations. It is adapted from the code described in the block
preconditioning chapter of the FENiCS book, by Kent-Andre Mardal
<kent-and@simula.no>.

The block structure is as follows,

       | A   B |
  AA = |       |,
       | C   0 |

and the preconditioner to be used is

        | A^  0   |
  BB^ = |         |,
        | 0  L^+M^|

with M defined as the scaled mass matrix and L as the Laplace operator. We use
the ML algebraic multigrid preconditioner for A, L, and M, and the CGN
iterative solver in order to get eigenvalue estimates for the preconditioned
systems.
"""
from __future__ import division
from __future__ import print_function

from dolfin import *
from block import *
from block.iterative import CGN
from block.algebraic.petsc import AMG
import numpy

set_log_level(30)

N=2
k_val=1

# Parse command-line arguments like "N=6" or "k_val=0.1"
import sys
for s in sys.argv[1:]:
    exec(s)

class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


BoundaryFunction = Expression(('std::abs(x[1]-1) < 1E-13 ? 1: 0', '0'), degree=1)


mesh = UnitSquareMesh(N,N)

V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)

f = Constant((0,0))
g = Constant(0)
k = Constant(k_val)
kinv = Constant(1.0/k_val)

v, u = TestFunction(V), TrialFunction(V)
q, p = TestFunction(Q), TrialFunction(Q)

a11 = (dot(u,v) + k*inner(grad(u),grad(v)))*dx
a12 = div(v)*p*dx
a21 = div(u)*q*dx
L1  = dot(f, v)*dx

bcs = block_bc([DirichletBC(V, BoundaryFunction, Boundary()), None], True)
AA = block_assemble([[a11, a12],
                     [a21,  0 ]])
bb = block_assemble([L1, 0])
bcs.apply(AA).apply(bb)

[[A, B],
 [C, _]] = AA

M = assemble(kinv*p*q*dx)
L = assemble(dot(grad(p),grad(q))*dx)

prec = block_mat([[AMG(A),      0     ],
                  [0,     AMG(L)+AMG(M)]])

xx = AA.create_vec()
xx.randomize()
AAinv = CGN(AA, precond=prec, initial_guess=xx, tolerance=1e-11, show=0)

x = AAinv*bb
e = AAinv.eigenvalue_estimates()

print("N=%d K=%.3g" % (N, sqrt(e[-1]/e[0])))
