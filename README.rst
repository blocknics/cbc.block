*cbc.block* is a python library for block operations in DOLFIN
(http://fenicsproject.org). The headline features are:

- Block operators may be defined from standard DOLFIN matrices and vectors::

    A = assemble(...); B = assemble(...); # etc
    AA = block_mat([[A,B], [C,D]])

- Preconditioners, inverses, and inner solvers are supported::

    AAprec = AA.scheme('gauss-seidel', inverse=ML)

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
DOLFIN python interface is required. For more details of use, I recommend
looking at the demos (start with demo/mixedpoisson.py), and the comments
therein.

Bugs, questions, contributions: Visit http://bitbucket.org/fenics-apps/cbc.block.

  The code is licensed under the GNU Lesser Public License, found in COPYING,
  version 2.1 or later. Some files under block/iterative/ use the BSD license,
  this is noted in the individual files.

Installation
------------
Using conda-forge,
```
# Create an environment

mamba create haznics
mamba activate haznics

# Install dependencies. To install fenics_ii or cbc.block from source,
# use "git clone https://..." followed by "pip install -e <dir>" instead.

mamba install fenics-dolfin quadpy
pip install "fenics_ii @ git+https://github.com/MiroK/fenics_ii"
pip install "cbc.block @ git+https://bitbucket.org/fenics-apps/cbc.block"

# Install haznics from source. Examples are in examples/haznics.

git clone https://github.com/HAZmathTeam/hazmath
cd hazmath
mamba install gcc gfortran make swig
make config shared=yes suitesparse=yes lapack=yes haznics=yes swig=yes
make install
cat >swig_files/setup.py <<-EOF
	from distutils.core import setup
	setup(name='haznics', py_modules=['haznics'])
EOF
python -m pip install -e swig_files
```

Publications
------------

1. K.-A. Mardal, and J. B. Haga (2012). *Block preconditioning of systems of PDEs.* In A. Logg, K.-A. Mardal, G. N. Wells et al. (ed) *Automated Solution of Differential Equations by the Finite Element Method,* Springer. doi:10.1007/978-3-642-23099-8, http://fenicsproject.org/book
