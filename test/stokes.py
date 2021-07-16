from __future__ import division
from __future__ import print_function
from past.utils import old_div
from dolfin import *
from block import *
from block.iterative import *
from block.algebraic.petsc import AMG 
import unittest


class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] <= DOLFIN_EPS  


class Stokes(unittest.TestCase): 
    def test(self):
        N = 8
        porder = 1
        vorder = 2
        alpha = 0

        mesh = UnitSquareMesh(N,N)

        V = VectorFunctionSpace(mesh, "CG", vorder)
        Q = FunctionSpace(mesh, "CG", porder)

        f = Constant((0,0))
        g = Constant(0)
        alpha = Constant(alpha)
        h = CellDiameter(mesh)

        v,u = TestFunction(V), TrialFunction(V)
        q,p = TestFunction(Q), TrialFunction(Q)

        a11 = inner(grad(v), grad(u))*dx
        a12 = div(v)*p*dx
        a21 = div(u)*q*dx
        a22 = -alpha*h*h*dot(grad(p), grad(q))*dx
        L1  = inner(v, f)*dx
        L2  = q*g*dx
        M1 = assemble(p*q*dx)

        bcs_exp = Expression(("-sin(x[1]*pi)", "0.0"), degree=3)
        bcs = DirichletBC(V, bcs_exp, Boundary())
        AA = block_assemble([[a11, a12], [a21, a22]])
        bb = block_assemble([L1, L2])

        block_bc([bcs], True).apply(AA).apply(bb)


        [[A, B],
         [C, D]] = AA

        BB = block_mat([[AMG(A),  0],
                        [0, AMG(M1)]])

        AAinv = MinRes(AA, precond=BB, tolerance=1e-8, show=0)
        x = AAinv * bb
        x.randomize()

        AAi = CGN(AA, precond=BB, initial_guess=x, tolerance=1e-8, maxiter=1000, show=0)
        AAi * bb

        e = AAi.eigenvalue_estimates()
        c =  sqrt(old_div(e[-1],e[0]))
        self.assertAlmostEqual(c, 4.403, places=2)


if __name__ == "__main__" :
    print("")
    print("Testing Stokes preconditioner")
    print("---------------------------------------------------------------------------")
    unittest.main()


