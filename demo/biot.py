from __future__ import division

"""This demo program shows the use of block preconditioners with Biot's consolidation equations.

The algebraic system to be solved can be written as

  BB^ AA [u p]^T = BB^ [b c]^T,

where AA is a 2x2 block system with a negative-definite (2,2) block D. The
system as a whole is symmetric, with C=B'.

       | A   B |
  AA = |       |,
       | C   D |

and BB^ approximates the inverse of the block operator

       | A   0 |
  BB = |       |,
       | 0   S |

where S=C*A^*B-D is (an approximation of) the pressure Schur complement. This
preconditioner is known as Generalized Jacobi. In this program, we use the
approximation S=C*diag(A)^*B-D for the Schur complement, ML approximations of
the inverses, and the SymmLQ iterative solver (suitable for indefinite
symmetric systems).

As an alternative solver (implemented below, but commented out), we can invert
AA directly using the Schur complement:

      | I    0 |   | A^  0 |   | I   A^*B |
AA^ = |        | x |       | x |          |,
      | C*A^ I |   | 0  -S^|   | 0   I    |

where the inverses are taken as exact. We note that this AA^ is equivalent to a
symmetric Gauss-Seidel block scheme, wherein the D block is replaced by -S.

We use conjugated gradient inner solvers for A^ and S^, preconditioned with the
ML approximations. Since the inverse is in principle exact, we do not need an
outer iterative solver, but nevertheless we use a single iteration of the
Richardson method just for reporting purposes.
"""

from block import *
from block.iterative import *
from block.algebraic.trilinos import *
from dolfin import *
from block.dolfin_util import *
import numpy

# Function spaces, elements

mesh = UnitSquare(16,16)
dim = mesh.topology().dim()

V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)

v, omega = TrialFunction(V), TestFunction(V)
q, phi   = TrialFunction(W), TestFunction(W)

u_prev = Function(V)
p_prev = Function(W)

#========
# Define forms, material parameters, boundary conditions, etc.

### Material parameters

lmbda = Constant(1e5)
mu    = Constant(1e5)

S = Constant(1e-2)
alpha = Constant(1.0)
Lambda = Constant( numpy.diag([.02]*(dim-1)+[.001]) )

t_n = Constant([0.0]*dim)
f_n = Constant([0.0]*dim)

dt = Constant(.02)
T = 0.1

Q = Constant(0)

def sigma(v):
    return 2.0*mu*sym(grad(v)) + lmbda*tr(grad(v))*Identity(dim)
def v_D(q):
    return -Lambda*grad(q)
def b(w,r):
    return - alpha * r * div(w)

a00 = inner(grad(omega), sigma(v)) * dx
a01 = b(omega,q) * dx
a10 = b(v,phi) * dx
a11 = -(S*phi*q - dt*inner(grad(phi),v_D(q))) * dx

L1 = b(u_prev,phi) * dx - (Q*dt + S*p_prev)*phi * dx

# Create boundary conditions.

boundary = BoxBoundary(mesh)

c = 0.25
h = mesh.hmin()
fluid_source_domain = compile_subdomains('{min}<x[0] && x[0]<{max} && {min}<x[1] && x[1]<{max}'
                                         .format(min=c-h, max=c+h))
topload_source      = Expression("-sin(2*t*pi)*sin(x[0]*pi/2)/3", t=0)

bc_u_bedrock        = DirichletBC(V,            [0]*dim,        boundary.bottom)
bc_u_topload        = DirichletBC(V.sub(dim-1), topload_source, boundary.top)
bc_p_drained_source = DirichletBC(W,            0,              fluid_source_domain)

bcs = [[bc_u_topload, bc_u_bedrock], [bc_p_drained_source]]

# Assemble the matrices
A = assemble(a00)
B = assemble(a01)
C = assemble(a10)
D = assemble(a11)
c = assemble(L1)

# Insert the matrices into blocks

AA = block_mat([[A, B],
                [C, D]])
bb = block_vec([0, c])

# Apply boundary conditions. Because just the right-hand side is modified later
# (in the time loop), and because the left-hand side is modified symmetrically,
# we set the save_A flag.

bcs = block_bc(bcs)
bcs.apply(AA, bb, save_A=True)

# Create preconditioner -- a generalised block Jacobi preconditioner, where the
# (2,2) approximates the pressure Schur complement. Since the ML preconditioner
# requires access to the matrix elements, we use the collapse() call to perform
# the necessary matrix algebra to convert the operator S to a single matrix.

Ap = ML(A)

S = C*InvDiag(A)*B-D
Sp = ML(collapse(S))

AApre = block_mat([[Ap, 0],
                   [0, -Sp]])


# Note: The preconditioner is indefinite, so this is a bit iffy. Most of the
# time MinRes works fine, but occasionally it fails -- fall back on the slower
# BiCGStab in those cases.
AAinv = MinRes(AA, precond=AApre, show=2, tolerance=1e-10)
AAinv_fallback = BiCGStab(AA, precond=AApre, show=2, tolerance=1e-10)

# An alternative could be to use an exact block decomposition of AAinv, like
# the following. Since the AApre we define is exact, it can be solved using
# AAinv=AApre, but we wrap it in a single-iteration Richardson solver as a
# simple way to get the time and residual printout. A single iteration of
# Richardson computes x=P*(b-A*x0), which equals P*b since the initial guess x0
# is zero (default).

#Ainv = ConjGrad(A, precond=Ap, name='A^', tolerance=1e-7)
#S = C*Ainv*B-D
#Sinv = ConjGrad(S, precond=Sp, name='S^', tolerance=1e-4)
#AApre = block_mat([[Ainv, B],
#                   [C, -Sinv]]).scheme('sgs')
#AAinv = Richardson(AA, precond=AApre, iter=1)

#=====================
# Set arguments to plot/save functions

#update.set_args(displacement={'mode': 'displacement', 'wireframe': True},
#                volumetric={'functionspace': FunctionSpace(mesh, "DG", 0)},
#                velocity={'functionspace': VectorFunctionSpace(mesh, "DG", 0)}
#                )

#=====================
# Time loop

t = 0.0
x = None
while t <= T:
    print "Time step %f" % t

    topload_source.t = t

    bb[0].zero()
    bb[1] = assemble(L1)
    bcs.apply(bb)

    try:
        x = AAinv * bb
    except ValueError:
        x = AAinv_fallback(initial_guess=x) * bb

    U,P = x
    u = Function(V, U)
    p = Function(W, P)

#    update(time=t,
#           displacement=u,
#           pressure=p,
#           velocity=v_D(p),
#           )

    u_prev.vector()[:] = U
    p_prev.vector()[:] = P
    t += float(dt)

#interactive()
print "Finished normally"
