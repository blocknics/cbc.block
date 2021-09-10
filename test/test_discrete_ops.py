from dolfin import *
from block.algebraic.hazmath import discrete_curl, discrete_gradient, Pcurl, Pdiv


def test_grad():
    mesh = UnitSquareMesh(3, 3)

    P1 = FunctionSpace(mesh, 'CG', 1)
    Ned = FunctionSpace(mesh, 'Nedelec 1st kind H(curl)', 1)

    f = Expression('x[0]+x[1]', degree=1)

    x, y = SpatialCoordinate(mesh)
    f_ = x + y
    f_P1 = interpolate(f, P1)

    G = discrete_gradient(mesh)

    x = G*f_P1.vector()
    grad_f = Function(Ned, x)

    e = inner(grad_f - grad(f_), grad_f - grad(f_))*dx

    assert sqrt(abs(assemble(e))) < 1E-14


def test_curl():
    mesh = UnitCubeMesh(1, 1, 1)
    V = FunctionSpace(mesh, 'Nedelec 1st kind H(curl)', 1)
    W = FunctionSpace(mesh, 'RT', 1)

    f = Expression(('x[0]+x[1]', '2*x[1]-x[0]', '-x[2]-x[1]-x[0]'), degree=1)

    x, y, z = SpatialCoordinate(mesh)
    f_ = as_vector((x+y, 2*y-x, -z-y-x))
    f_V = interpolate(f, V)

    C = discrete_curl(mesh)

    curl_f1 = Function(W)
    curl_f1.vector()[:] = C*f_V.vector()

    e = inner(curl(f_) - curl_f1, curl(f_) - curl_f1)*dx
    assert sqrt(abs(assemble(e))) < 1E-14
    

def test_Pcurl():
    mesh = UnitCubeMesh(1, 1, 1)
    Ned = FunctionSpace(mesh, 'Nedelec 1st kind H(curl)', 1)
    P1 = VectorFunctionSpace(mesh, 'CG', 1)

    f = Expression(('x[0]+x[1]', '2*x[1]-x[0]', '-x[2]-x[1]-x[0]'), degree=1)

    u1 = interpolate(f, P1)
    u2 = interpolate(f, Ned)

    Pc = Pcurl(mesh)

    u1_int = Function(Ned)
    u1_int.vector()[:] = (Pc * u1.vector())[:]

    e = inner(u1_int - u2, u1_int - u2) * dx
    assert sqrt(abs(assemble(e))) < 1E-14


def test_Pdiv():
    mesh = UnitCubeMesh(1, 1, 1)
    RT = FunctionSpace(mesh, 'Raviart-Thomas', 1)
    P1 = VectorFunctionSpace(mesh, 'CG', 1)

    f = Expression(('x[0]+x[1]', '2*x[1]-x[0]', '-x[2]-x[1]-x[0]'), degree=1)

    u1 = interpolate(f, P1)
    u2 = interpolate(f, RT)

    Pd = Pdiv(mesh)

    u1_int = Function(RT)
    u1_int.vector()[:] = (Pd * u1.vector())[:]

    e = inner(u1_int - u2, u1_int - u2) * dx
    assert sqrt(abs(assemble(e))) < 1E-14
