*cbc.block* is a python library for block operations in DOLFIN
(http://fenicsproject.org). The headline features are:

- Block operators may be defined from standard DOLFIN matrices and vectors:

    ```
    A = assemble(...); B = assemble(...); # etc
    AA = block_mat([[A,B], [C,D]])
    ```

- Preconditioners, inverses, and inner solvers are supported:

    ```
    AAprec = AA.scheme('gauss-seidel', inverse=ML)
    ```

- A good selection of iterative solvers:

    ```
    AAinv = SymmLQ(AA, precond=AAprec)
    x = AAinv*b
    ```

- Matrix algebra is supported both through composition of operators... :

    ```
    S = C*ILU(A)*B-D
    Sprec = ConjGrad(S)
    ```
    ...and through explicit matrix calculation via PyTrilinos:
  
    ```
    S = C*InvDiag(A)*B-D
    Sprec = ML(collapse(S))
    ```
  
There is no real documentation apart from the python doc-strings, but an
(outdated) introduction is found in
[doc/blockdolfin.pdf](https://github.com/blocknics/cbc.block/blob/master/doc/blockdolfin.pdf).
Familiarity with the DOLFIN python interface is required. For more details of
use, I recommend looking at the demos (start with demo/mixedpoisson.py), and
the comments therein.

Bugs, questions, contributions: Visit <http://github.com/blocknics/cbc.block>.

> The code is licensed under the GNU Lesser Public License, found in COPYING,
> version 2.1 or later. Some files under block/iterative/ use the BSD license,
> this is noted in the individual files.

Status (master branch)
----------------------
![Regression test status](https://github.com/blocknics/cbc.block/actions/workflows/test.yaml/badge.svg)

[Full test coverage report](https://blocknics.github.io/cbc.block/htmlcov/)

Installation
------------

The fenics-dolfin package is available from conda-forge, or preinstalled on
Docker images from <https://quay.io/organization/fenicsproject>. The recipe
below assumes that you have an existing conda-forge installation; to install on
Docker images, skip the first few steps.

```
# Create an environment

conda create cbc-block
conda activate cbc-block

# Install base and dolfin (and optionally pytrilinos, but it doesn't seem to have a compatible version atm)

conda install pip fenics-dolfin scipy

# Install haznics from source. Examples are in examples/haznics.
# This can be skipped, but some demos will fail to run.

HAZ_VER=1.0.1
git clone --branch v${HAZ_VER} --depth 1 https://github.com/HAZmathTeam/hazmath
cd hazmath

conda install compilers c-compiler cxx-compiler fortran-compiler cmake>=3.15 make swig
sed -i -e '/cmake_minimum_required/s/3.12/3.15/' CMakeLists.txt
make config shared=yes suitesparse=yes lapack=yes haznics=yes swig=yes
make install
cp -a swig_files haznics
mv haznics/haznics.py haznics/__init__.py
cat >setup.py <<-EOF
	from distutils.core import setup
	setup(name='haznics', version='${HAZ_VER}', packages=['haznics'],
          package_data={'haznics': ['_haznics.so']})
EOF
python -m pip install .

# Install cbc.block itself. To install from source,
# use "git clone https://..." followed by "pip install -e <dir>[haznics]" instead.

pip install "cbc.block[haznics] @ git+https://github.com/blocknics/cbc.block"
````

Publications
------------

1. K.-A. Mardal and J. B. Haga (2012). *Block preconditioning of systems of PDEs.* In A. Logg, K.-A. Mardal, G. N. Wells et al. (ed) *Automated Solution of Differential Equations by the Finite Element Method,* Springer. doi:10.1007/978-3-642-23099-8_35, <http://fenicsproject.org/book>
2. A. Budisa, X. Hu, M. Kuchta, K.-A. Mardal and L. Zikatanov (2022). *HAZniCS — Software Components for Multiphysics Problems.* doi: 10.48550/ARXIV.2210.13274, <https://arxiv.org/abs/2210.13274>
