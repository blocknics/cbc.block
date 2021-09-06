*cbc.block* is a python library for block operations in DOLFIN
(http://fenicsproject.org). The headline features are:

- Block operators may be defined from standard DOLFIN matrices and vectors::

    A = assemble(...); B = assemble(...); # etc
    AA = block_mat([[A,B], [C,D]])

- Preconditioners, inverses, and inner solvers are supported::

    AAprec = AA.scheme('gauss-seidel', inverse=ML)

- Currently two packages of preconditioners are available: PETSc/Hypre and HAZmath::

    from block.algebraic.petsc import ML
    from block.algebraic.hazmath import AMG

    AAprec = block_mat([[ML(A), 0], [ 0, AMG(D)]])

- A good selection of iterative solvers::

    AAinv = SymmLQ(AA, precond=AAprec)
    x = AAinv*b

- Matrix algebra is supported both through composition of operators... ::

    S = C*ILU(A)*B-D
    Sprec = ConjGrad(S)

  ...and through explicit matrix calculation via PyTrilinos::

    S = C*InvDiag(A)*B-D
    Sprec = ML(collapse(S))

There is no real documentation apart from the python doc-strings, but an
(outdated) introduction is found in doc/blockdolfin.pdf. Familiarity with the
DOLFIN python interface is required. Checkout also the
publications below. For more details of use, we recommend
looking at the demos (start with demo/poisson.py), and the comments
therein.

Bugs, questions, contributions: Visit http://bitbucket.org/fenics-apps/cbc.block.

  The code is licensed under the GNU General Public License, found in COPYING,
  version 3 or later. Some files under block/iterative/ use the BSD license,
  this is noted in the individual files.


Contributors: Joachim Berdal Haga, Kent-Andre Mardal, Martin Sandve Alnæs, Magne Nordaas, Miroslav Kuchta, Cécile Daversin-Catty, Ana Budiša.

Publications
------------

1. K.-A. Mardal, and J. B. Haga (2012). *Block preconditioning of systems of PDEs.* In A. Logg, K.-A. Mardal, G. N. Wells et al. (ed) *Automated Solution of Differential Equations by the Finite Element Method,* Springer. doi:10.1007/978-3-642-23099-8, http://fenicsproject.org/book
