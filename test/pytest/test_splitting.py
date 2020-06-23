from builtins import map
from dolfin import *
from block import *
from block.splitting import split_form

def test_split_form():
    # mixed form
    P2 = VectorElement("Lagrange", triangle, 2)
    P1 = FiniteElement("Lagrange", triangle, 1)
    TH = MixedElement([P2, P1])

    mesh = UnitSquareMesh(16, 16)

    W = FunctionSpace(mesh, TH)

    h = Function(W)
    f, g = h.split()
    # randomize right hand side
    from numpy.random import rand
    h.vector()[:] = rand(h.vector().local_size())

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    a_mixed = inner(grad(u), grad(v)) * dx \
            + p * div(v) * dx + q * div(u) * dx

    L_mixed = inner(f, v) * dx + g * q * dx

    # blocked form
    V, Q = [sub_space.collapse() for sub_space in W.split()]
    u, p = list(map(TrialFunction, [V, Q]))
    v, q = list(map(TestFunction, [V, Q]))

    from numpy import array
    a_blocked = array([[inner(grad(u), grad(v)) * dx, p * div(v) * dx],
                       [             q * div(u) * dx,               0]])

    L_blocked = array([inner(f, v) * dx, g * q * dx])

    # split blocked form
    a_split = split_form(a_mixed)
    L_split = split_form(L_mixed)

    # assemble each matrix
    parameters["form_compiler"]["quadrature_degree"] = 6
    A00_blocked = assemble(a_blocked[0,0])
    A01_blocked = assemble(a_blocked[0,1])
    A10_blocked = assemble(a_blocked[1,0])

    A00_split = assemble(a_split[0,0])
    A01_split = assemble(a_split[0,1])
    A10_split = assemble(a_split[1,0])
    A11_split = assemble(a_split[1,1])

    # assemble vectors
    b0_blocked = assemble(L_blocked[0])
    b1_blocked = assemble(L_blocked[1])

    b0_split = assemble(L_split[0])
    b1_split = assemble(L_split[1])

    # check equality
    def equal(A, B):
        AA = A.copy()
        if isinstance(AA, GenericVector):
           AA.axpy(-1, B)
        else:
            AA.axpy(-1, B, True)

        return AA.norm("linf") < DOLFIN_EPS * max(B.norm("linf"), B.norm("linf"))

    assert equal(A00_blocked, A00_split)
    assert equal(A01_blocked, A01_split)
    assert equal(A10_blocked, A10_split)
    assert A11_split.norm("linf") == 0

    assert equal(b0_blocked, b0_split)
    assert equal(b1_blocked, b1_split)
