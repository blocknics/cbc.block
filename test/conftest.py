from types import SimpleNamespace
import dolfin as df
import pytest

@pytest.fixture(scope='module')
def poisson():
    mesh = df.UnitSquareMesh(16,16)
    V = df.FunctionSpace(mesh, "CG", 1)
    f = df.Expression("sin(pi*x[0])", degree=2)
    u, v = df.TrialFunction(V), df.TestFunction(V)

    a = u*v*df.dx + df.dot(df.grad(u), df.grad(v))*df.dx
    L = f*v*df.dx

    A = df.assemble(a)
    b = df.assemble(L)

    u = df.Function(V)
    df.solve(A, u.vector(), b)

    return SimpleNamespace(A=A, b=b, V=V, x=u.vector())
