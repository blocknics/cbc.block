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
[doc/blockdolfin.pdf](https://github.com/fenics-apps/cbc.block/blob/jobh/master/doc/blockdolfin.pdf).
Familiarity with the DOLFIN python interface is required. For more details of
use, I recommend looking at the demos (start with demo/mixedpoisson.py), and
the comments therein.

Bugs, questions, contributions: Visit <http://github.com/fenics-apps/cbc.block>.

> The code is licensed under the GNU Lesser Public License, found in COPYING,
> version 2.1 or later. Some files under block/iterative/ use the BSD license,
> this is noted in the individual files.

Status (master branch)
----------------------
![Regression test status](https://github.com/fenics-apps/cbc.block/actions/workflows/test.yaml/badge.svg?branch=master)

[Full test coverage report](https://fenics-apps.github.io/cbc.block/htmlcov/)

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
pip install "cbc.block @ git+https://github.com/fenics-apps/cbc.block"

# Install haznics from source. Examples are in examples/haznics.
# Note, no branch or tag is currently compatible with cbc.block (master
# is too unstable, v1.0.0 is too old), so download a specific revision.

haz_rev=a2d5267b8c26dcc0fbfd75ff795cc9115d2331eb
curl -L -o hazmath.zip https://github.com/HAZmathTeam/hazmath/archive/${haz_rev}.zip
unzip hazmath.zip; rm hazmath.zip
cd hazmath-${haz_rev}

mamba install compilers c-compiler cxx-compiler fortran-compiler cmake>=3.15 make swig
sed -i -e '/cmake_minimum_required/s/3.12/3.15/' CMakeLists.txt
make config shared=yes suitesparse=yes lapack=yes haznics=yes swig=yes
make install
cp -a swig_files haznics
mv haznics/haznics.py haznics/__init__.py
cat >setup.py <<-EOF
	from distutils.core import setup
	setup(name='haznics', packages=['haznics'],
package_data={'haznics': ['_haznics.so']})
EOF
python -m pip install .
````

Publications
------------

1. K.-A. Mardal, and J. B. Haga (2012). *Block preconditioning of systems of PDEs.* In A. Logg, K.-A. Mardal, G. N. Wells et al. (ed) *Automated Solution of Differential Equations by the Finite Element Method,* Springer. doi:10.1007/978-3-642-23099-8, <http://fenicsproject.org/book>
