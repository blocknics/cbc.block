# (pytorch) mirok@evalApply:demo (master)*$ python hodge.py "N=16; dim=3"
"""This demo shows the use of a non-trivial block preconditioner for the Hodge
equations namely the curl-curl multigrid is employed for the Schur complement 
preconditioner. It is adapted from the code described in the block preconditioning
chapter of the FENiCS book, by Kent-Andre Mardal <kent-and@simula.no>.

The block structure is as follows,

       | A   B |
  AA = |       |,
       | C  -D |

where C=B' and D is positive definite; hence, the system as a whole is
symmetric indefinite.

The block preconditioner is based on an approximation of the Schur complement
of the (0,0) block, L=D+B*A^*C:

        | A  0 |
  BB^ = |      |,
        | 0  S |

"""

from __future__ import division
from __future__ import print_function

from dolfin import *
from block import *
from block.iterative import *
from block.algebraic.hazmath import AMG, HXCurl

set_log_level(30)

N = 4
dim = 3
# Parse command-line arguments like "N=6"
import sys
for s in sys.argv[1:]:
    exec(s)

mesh = {2: UnitSquareMesh, 3: UnitCubeMesh}[dim](*(N, )*dim)

V = FunctionSpace(mesh, "N1curl", 1)
Q = FunctionSpace(mesh, "CG", 1)

v,u = TestFunction(V), TrialFunction(V)
q,p = TestFunction(Q), TrialFunction(Q)

A = assemble(dot(u,v)*dx + dot(curl(v), curl(u))*dx)
B = assemble(dot(grad(p),v)*dx)
C = assemble(dot(grad(q),u)*dx)
D = assemble(p*q*dx)
E = assemble(p*q*dx + dot(grad(p),grad(q))*dx)

AA = block_mat([[A,  B],
                [C, -D]])

gdim = mesh.geometry().dim()
b0 = assemble(inner(v, Constant((1, )*gdim))*dx)
              
b1 = assemble(inner(q, Constant(2))*dx)
bb = block_vec([b0, b1])

prec = block_mat([[HXCurl(A, V),  0  ],
                  [0,            AMG(E)]])
AAinv = MinRes(AA, precond=prec, tolerance=1e-9, maxiter=2000, show=2)
[Uh, Ph] = AAinv*bb

if MPI.size(mesh.mpi_comm()) == 1 and gdim == 2:
    import matplotlib.pyplot as plt

    plt.subplot(121)
    plot(Function(V, Uh))

    plt.subplot(122)    
    plot(Function(Q,  Ph))

    plt.show()
