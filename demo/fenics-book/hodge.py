from __future__ import division

"""This demo shows the use of a non-trivial block preconditioner for the Hodge
equations. It is adapted from the code described in the block preconditioning
chapter of the FENiCS book, by Kent-Andre Mardal <kent-and@simula.no>.

The block structure is as follows,

       | A   B |
  AA = |       |,
       | C  -D |

where C=B' and D is positive definite; hence, the system as a whole is
symmetric indefinite.

The block preconditioner is based on an approximation of the Schur complement
of the (1,1) block, L=A+B*D^*C:

        | L^  0 |
  BB^ = |       |,
        | 0   D^|

where the L block is formed explicitly by matrix multiplication (using the
collapse() method), and AMG is used for the single-block preconditioners.  The
CGN iterative solver in order to get eigenvalue estimates for the
preconditioned systems.
"""

from dolfin import *
from block import *
from block.iterative import *
from block.algebraic.petsc import *

dolfin.set_log_level(30)

N = 4

# Parse command-line arguments like "N=6"
import sys
for s in sys.argv[1:]:
    exec(s)

mesh = UnitCubeMesh(N,N,N)

V = FunctionSpace(mesh, "N1curl", 1)
Q = FunctionSpace(mesh, "CG", 1)

v,u = TestFunction(V), TrialFunction(V)
q,p = TestFunction(Q), TrialFunction(Q)

A = assemble(dot(u,v)*dx + dot(curl(v), curl(u))*dx)
B = assemble(dot(grad(p),v)*dx)
C = assemble(dot(grad(q),u)*dx)
D = assemble(p*q*dx)
E = assemble(dot(grad(p),grad(q))*dx)

AA = block_mat([[A,  B],
                [C, -D]])
bb = block_vec([0,0])

L = collapse(A+B*InvDiag(D)*C)

x = AA.create_vec(dim=1)
x.randomize()
bb.allocate(AA, dim=0)

Linv = CGN(L, precond=AMG(L), initial_guess=x[0], tolerance=1e-9, maxiter=2000, show=0)
Linv * bb[0]
e = Linv.eigenvalue_estimates()
K_P2L = sqrt(e[-1]/e[0])

Linv = CGN(A, precond=AMG(A), initial_guess=x[0], tolerance=1e-9, maxiter=2000, show=0)
Linv * bb[0]
e = Linv.eigenvalue_estimates()
K_P1A = sqrt(e[-1]/e[0])

# Note april-2014: PETSc-AMG fails -- AMG(E) not positive definite. Find appropriate smoother?
prec = block_mat([[AMG(A),  0  ],
                  [0,    ILU(E)]])
AAinv = CGN(AA, precond=prec, initial_guess=x, tolerance=1e-9, maxiter=2000, show=0)
AAinv*bb
e = AAinv.eigenvalue_estimates()
K_B1AA = sqrt(e[len(e)-1]/e[0])

prec = block_mat([[AMG(L),  0  ],
                  [0,    ILU(D)]])
AAinv = CGN(AA, precond=prec, initial_guess=x, tolerance=1e-9, maxiter=2000, show=0)
AAinv*bb
e = AAinv.eigenvalue_estimates()
K_B2AA = sqrt(e[len(e)-1]/e[0])


print 'N=%d P1A=%.3g P2L=%.3g B1AA=%.3g   B2AA=%.3g' % (N, K_P1A, K_P2L, K_B1AA,  K_B2AA)



