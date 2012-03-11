from __future__ import division

"""Utility functions for plotting, boundaries, etc."""

import os
import dolfin as d
import time as timer

class BoxBoundary(object):
    def __init__(self, mesh):
        c = mesh.coordinates()
        self.c_min, self.c_max = map(d.MPI.min, c.min(0)), map(d.MPI.max, c.max(0))
        dim = len(self.c_min)

        sd = self._compile(west  = self._boundary(0, self.c_min) if dim>1 else '0',
                           east  = self._boundary(0, self.c_max) if dim>1 else '0',
                           south = self._boundary(1, self.c_min) if dim>2 else '0',
                           north = self._boundary(1, self.c_max) if dim>2 else '0',
                           bottom= self._boundary(dim-1, self.c_min),
                           top   = self._boundary(dim-1, self.c_max),
                           ew    = self._boundary(0) if dim>1 else '0',
                           ns    = self._boundary(1) if dim>1 else '0',
                           tb    = self._boundary(dim-1),
                           all   = 'on_boundary')
        for name,subdomain in sd:
            setattr(self, name, subdomain)

    def _boundary(self, idx, coords=None):
        if coords is not None:
            return 'on_boundary && near(x[{idx}], {coord})' \
                .format(idx=idx, coord=coords[idx])
        else:
            return 'on_boundary && (near(x[{idx}], {min}) || near(x[{idx}], {max}))' \
                .format(idx=idx, min=self.c_min[idx], max=self.c_max[idx])

    def _compile(self, **kwargs):
        # Make sure all expressions sent to compile_subdomains are different
        expr_to_code = {}
        for expr in kwargs.values():
            expr_to_code[expr] = None

        compiled = d.compile_subdomains(expr_to_code.keys())
        for i, expr in enumerate(expr_to_code.keys()):
            expr_to_code[expr] = compiled[i]

        return [(name, expr_to_code[expr]) for name, expr in kwargs.items()]

class update():
    """Plot and save given functional(s). Example:
    u = problem.solve()
    update.set_args(displacement={'mode': 'displacement'})
    update(displacement=u, volumetric=tr(sigma(u)))
    """
    files = {}
    plots = {}
    kwargs = {}
    projectors = {}
    functions = {}

    def _extract_function_space(self, expression, mesh=None):
        """Try to extract a suitable function space for projection of
        given expression. Copied from dolfin/fem/projection.py"""
        import ufl

        # Extract functions
        functions = ufl.algorithms.extract_coefficients(expression)

        # Extract mesh from functions
        if mesh is None:
            for f in functions:
                if isinstance(f, d.Function):
                    mesh = f.function_space().mesh()
                    if mesh is not None:
                        break
                    if mesh is None:
                        raise RuntimeError, "Unable to project expression, no suitable mesh."

        # Create function space
        shape = expression.shape()
        if shape == ():
            V = d.FunctionSpace(mesh, "CG", 1)
        elif len(shape) == 1:
            V = d.VectorFunctionSpace(mesh, "CG", 1, dim=shape[0])
        elif len(shape) == 2:
            V = d.TensorFunctionSpace(mesh, "CG", 1, shape=shape)
        else:
            raise RuntimeError, "Unable to project expression, unhandled rank, shape is %s." % (shape,)

        return V

    def project(self, f, name, V, mesh=None):
        if V is None:
            # If trying to project an Expression
            if isinstance(f, d.Expression):
                if isinstance(mesh, cpp.Mesh):
                    V = FunctionSpaceBase(mesh, v.ufl_element())
                else:
                    raise TypeError, "expected a mesh when projecting an Expression"
            else:
                V = self._extract_function_space(f, mesh)
        key = str(V)
        v = d.TestFunction(V)
        if not key in self.projectors:
            # Create mass matrix
            u = d.TrialFunction(V)
            a = d.inner(v,u) * d.dx
            solver = d.LinearSolver("cg", "none")
            solver.set_operator(d.assemble(a))
            solver.parameters['preconditioner']['reuse'] = True
            self.projectors[key] = solver
        # Use separate function objects for separate quantities, since this
        # object is used as key by viper
        if not name in self.functions:
            self.functions[name] = d.Function(V)

        solver, Pf = self.projectors[key], self.functions[name]
        b = d.assemble(d.inner(v,f) * d.dx)

        # Solve linear system for projection
        solver.solve(Pf.vector(), b)

        return Pf


    def set_args(self, **kwargs):
        """Set additional kwargs to pass to plot for a given name.

        In addition to the kwargs for plot, these are accepted:
        'plot' (bool)                   -- plot to screen [True]
        'save' (bool)                   -- save to file [True]
        'functionspace' (FunctionSpace) -- space to project to [CG(1)]"""
        self.kwargs.update(kwargs)

    def save_to_file(self, name, data, time):
        if not os.path.exists('data'):
            os.mkdir('data')
        if not name in self.files:
            self.files[name] = d.File('data/%s.pvd'%name)
        if time is not None:
            self.files[name] << (data, time)
        else:
            self.files[name] << data

    def plot(self, name, title, data, time):
        kwargs = self.kwargs.get(name, {})
        if not name in self.plots:
            self.plots[name] = d.plot(data, title=title, size=(400,400),
                                      axes=True, warpscalar=False,
                                      **kwargs)
        else:
            self.plots[name].update(data, title=title, **kwargs)

    def __call__(self, time=None, postfix="", **functionals):
        for name,func in sorted(functionals.iteritems()):
            args = self.kwargs.get(name, {})
            if 'functionspace' in args or not isinstance(func, d.Function):
                func = self.project(func, name, args.get('functionspace'))
            if hasattr(func, 'rename'):
                func.rename(name+postfix, name+postfix)
            if args.get('plot', True):
                self.plot(name, name+postfix, func, time)
            if args.get('save', True):
                self.save_to_file(name+postfix, func, time)

update = update() # singleton
