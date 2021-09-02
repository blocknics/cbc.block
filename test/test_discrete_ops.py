from dolfin import *
from block.algebraic.petsc import HypreADS


def test_grad():
    mesh = UnitSquareMesh(3, 3)

    P1 = FunctionSpace(mesh, 'CG', 1)
    Ned = FunctionSpace(mesh, 'Nedelec 1st kind H(curl)', 1)

    f = Expression('x[0]+x[1]', degree=1)

    x, y = SpatialCoordinate(mesh)
    f_ = x + y
    f_P1 = interpolate(f, P1)

    G = PETScMatrix(HypreADS.discrete_gradient(mesh))

    x = G*f_P1.vector()
    grad_f = Function(Ned, x)

    e = inner(grad_f - grad(f_), grad_f - grad(f_))*dx

    assert sqrt(abs(assemble(e))) < 1E-14


def test_curl():
    mesh = UnitCubeMesh(2, 3, 4)
    V = FunctionSpace(mesh, 'Nedelec 1st kind H(curl)', 1)
    W = FunctionSpace(mesh, 'RT', 1)

    f = Expression(('x[0]+x[1]', '2*x[1]-x[0]', '-x[2]-x[1]-x[0]'), degree=1)

    x, y, z = SpatialCoordinate(mesh)
    f_ = as_vector((x+y, 2*y-x, -z-y-x))
    f_V = interpolate(f, V)

    C = PETScMatrix(HypreADS.discrete_curl(mesh))

    curl_f1 = Function(W)
    curl_f1.vector()[:] = C*f_V.vector()

    e = inner(curl(f_) - curl_f1, curl(f_) - curl_f1)*dx
    assert sqrt(abs(assemble(e))) < 1E-14
    
    q = TestFunction(W)
    n = FacetNormal(mesh)
    x = assemble(inner(n('+'), q('+'))*dS + inner(n, q)*ds)

test_grad()
test_curl()
