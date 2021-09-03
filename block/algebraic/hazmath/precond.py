from block.block_base import block_base
from builtins import str
from petsc4py import PETSc
from scipy.sparse import csr_matrix
import dolfin as df
import numpy as np
import haznics

# ------------------------------------------------------------------- #
# --------------           auxiliary functions        --------------- #
# ------------------------------------------------------------------- #


def PETSc_to_dCSRmat(A):
    """
    Change data type for matrix
    (dolfin PETScMatrix or GenericMatrix to dCSRmat pointer)
    """
    if not isinstance(A, df.PETScMatrix):
        A = df.as_backend_type(A)
    petsc_mat = A.mat()

    # NB! store copies for now
    csr0 = petsc_mat.getValuesCSR()[0]
    csr1 = petsc_mat.getValuesCSR()[1]
    csr2 = petsc_mat.getValuesCSR()[2]

    return haznics.create_matrix(csr2, csr1, csr0)


# copied from block/algebraic/petsc/
def discrete_gradient(mesh):
    """ P1 to Ned1 map """
    Ned = df.FunctionSpace(mesh, 'Nedelec 1st kind H(curl)', 1)
    P1 = df.FunctionSpace(mesh, 'CG', 1)

    return df.as_backend_type(df.DiscreteOperators.build_gradient(Ned, P1))


def discrete_curl(mesh):
    """ Ned1 to RT1 map """
    assert mesh.geometry().dim() == 3
    assert mesh.topology().dim() == 3

    Ned = df.FunctionSpace(mesh, 'Nedelec 1st kind H(curl)', 1)
    RT = df.FunctionSpace(mesh, 'RT', 1)

    RT_f2dof = np.array(RT.dofmap().entity_dofs(mesh, 2))
    Ned_e2dof = np.array(Ned.dofmap().entity_dofs(mesh, 1))

    # Facets in terms of edges
    mesh.init(2, 1)
    f2e = mesh.topology()(2, 1)

    rows = np.repeat(RT_f2dof, 3)
    cols = Ned_e2dof[f2e()]
    vals = np.tile(np.array([-2, 2, -2]), RT.dim())

    Ccsr = csr_matrix((vals, (rows, cols)), shape=(RT.dim(), Ned.dim()))

    C = PETSc.Mat().createAIJ(comm=df.MPI.comm_world,
                              size=Ccsr.shape,
                              csr=(Ccsr.indptr, Ccsr.indices, Ccsr.data))

    return df.PETScMatrix(C)


# NB: this is only 3d!
def Pdiv(mesh):
    """ nodes to faces map """
    RT = df.FunctionSpace(mesh, 'Raviart-Thomas', 1)
    P1 = df.VectorFunctionSpace(mesh, 'CG', 1)

    RT_f2dof = np.array(RT.dofmap().entity_dofs(mesh, 2))
    P1_n2dof_x = np.array(P1.sub(0).dofmap().entity_dofs(mesh, 0))
    P1_n2dof_y = np.array(P1.sub(1).dofmap().entity_dofs(mesh, 0))
    P1_n2dof_z = np.array(P1.sub(2).dofmap().entity_dofs(mesh, 0))

    # Facets in terms of nodes
    mesh.init(2, 0)
    f2n = mesh.topology()(2, 0)
    coordinates = mesh.coordinates()

    rows_x, rows_y, rows_z = np.repeat(RT_f2dof, 3), np.repeat(RT_f2dof, 3), np.repeat(RT_f2dof, 3)
    cols_x, cols_y, cols_z = P1_n2dof_x[f2n()], P1_n2dof_y[f2n()], P1_n2dof_z[f2n()]
    vals_x, vals_y, vals_z = np.zeros(3 * RT.dim()), np.zeros(3 * RT.dim()), np.zeros(3 * RT.dim())

    for facet, row in enumerate(RT_f2dof):
        vertices = f2n(facet)
        n1, n2, n3 = coordinates[vertices]

        facet_normal = np.cross(n3 - n1, n2 - n1)

        indices = np.arange(3) + 3 * facet
        vals_x[indices] = facet_normal[0] / 3
        vals_y[indices] = facet_normal[1] / 3
        vals_z[indices] = facet_normal[2] / 3

    rows = np.concatenate((rows_x, rows_y, rows_z))
    cols = np.concatenate((cols_x, cols_y, cols_z))
    vals = np.concatenate((vals_x, vals_y, vals_z))

    # assemble Pdiv as from xyz components
    Pdivcsr = csr_matrix((vals, (rows, cols)), shape=(RT.dim(), P1.dim()))

    Pdiv = PETSc.Mat().createAIJ(comm=df.MPI.comm_world,
                                 size=Pdivcsr.shape,
                                 csr=(Pdivcsr.indptr, Pdivcsr.indices, Pdivcsr.data))

    return df.PETScMatrix(Pdiv)


def Pcurl(mesh):
    """ nodes to edges map """
    assert mesh.geometry().dim() == 3
    assert mesh.topology().dim() == 3

    Ned = df.FunctionSpace(mesh, 'Nedelec 1st kind H(curl)', 1)
    P1 = df.VectorFunctionSpace(mesh, 'CG', 1)

    Ned_e2dof = np.array(Ned.dofmap().entity_dofs(mesh, 1))
    P1_n2dof_x = np.array(P1.sub(0).dofmap().entity_dofs(mesh, 0))
    P1_n2dof_y = np.array(P1.sub(1).dofmap().entity_dofs(mesh, 0))
    P1_n2dof_z = np.array(P1.sub(2).dofmap().entity_dofs(mesh, 0))

    # Facets in terms of edges
    mesh.init(1, 0)
    e2n = mesh.topology()(1, 0)
    coordinates = mesh.coordinates()

    rows_x, rows_y, rows_z = np.repeat(Ned_e2dof, 2), np.repeat(Ned_e2dof, 2), np.repeat(Ned_e2dof, 2)
    cols_x, cols_y, cols_z = P1_n2dof_x[e2n()], P1_n2dof_y[e2n()], P1_n2dof_z[e2n()]
    vals_x, vals_y, vals_z = np.zeros(2 * Ned.dim()), np.zeros(2 * Ned.dim()), np.zeros(2 * Ned.dim())

    for edge, row in enumerate(Ned_e2dof):
        vertices = e2n(edge)
        edge_tangent = coordinates[vertices[1]] - coordinates[vertices[0]]

        indices = np.arange(2) + 2 * edge
        vals_x[indices] = edge_tangent[0] / 2
        vals_y[indices] = edge_tangent[1] / 2
        vals_z[indices] = edge_tangent[2] / 2

    rows = np.concatenate((rows_x, rows_y, rows_z))
    cols = np.concatenate((cols_x, cols_y, cols_z))
    vals = np.concatenate((vals_x, vals_y, vals_z))

    # assemble Pcurl from xyz components
    Pcurlcsr = csr_matrix((vals, (rows, cols)), shape=(Ned.dim(), P1.dim()))

    Pcurl = PETSc.Mat().createAIJ(comm=df.MPI.comm_world,
                                  size=Pcurlcsr.shape,
                                  csr=(Pcurlcsr.indptr, Pcurlcsr.indices, Pcurlcsr.data))

    return df.PETScMatrix(Pcurl)


# ------------------------------------------------------------------- #
# --------------             preconditioners          --------------- #
# ------------------------------------------------------------------- #


class Precond(block_base):
    """
    Class of general preconditioners from HAZmath using SWIG

    """

    def __init__(self, A, prectype=None, parameters=None, precond=None):
        # haznics.dCSRmat* type (assert?)
        self.A = A
        # python dictionary of parameters
        self.parameters = parameters if parameters else {}

        # init and set preconditioner (precond *)
        if precond:
            self.precond = precond
        else:
            import warnings
            warnings.warn(
                "!! Preconditioner not specified !! Creating default UA-AMG "
                "precond... ",
                RuntimeWarning)
            # change data type for the matrix (to dCSRmat pointer)
            A_ptr = PETSc_to_dCSRmat(A)

            # initialize amg parameters (AMG_param pointer)
            amgparam = haznics.amg_param_alloc(1)

            # print (relevant) amg parameters
            haznics.param_amg_print(amgparam)

            self.precond = haznics.create_precond(A_ptr, amgparam)

            # if fail, setup returns null
            if not precond:
                raise RuntimeError(
                    "AMG levels failed to set up (null pointer returned) ")

        # preconditioner type (string)
        self.prectype = prectype if prectype else "AMG"

    def matvec(self, b):
        if not isinstance(b, df.GenericVector):
            return NotImplemented

        x = self.A.create_vec(dim=1)
        if len(x) != len(b):
            raise RuntimeError('incompatible dimensions for matvec, %d != %d'
                               % (len(x), len(b)))

        # convert rhs and dx to numpy arrays
        b_np = b[:]
        x_np = x[:]

        # apply the preconditioner (solution dx saved in x_np)
        haznics.apply_precond(b_np, x_np, self.precond)

        # convert dx to GenericVector
        x.set_local(x_np)

        return x

    # noinspection PyMethodMayBeStatic
    def down_cast(self):
        return NotImplemented

    def __str__(self):
        return '<%s prec of %s>' % (self.__class__.__name__, str(self.A))


class AMG(Precond):
    """
    AMG preconditioner from the HAZmath Library with SWIG

    """

    def __init__(self, A, parameters=None):
        # change data type for the matrix (to dCSRmat pointer)
        A_ptr = PETSc_to_dCSRmat(A)

        # initialize amg parameters (AMG_param pointer)
        amgparam = haznics.AMG_param()

        # set extra amg parameters
        if parameters:
            haznics.param_amg_set_dict(parameters, amgparam)

        # print (relevant) amg parameters
        haznics.param_amg_print(amgparam)

        # set AMG preconditioner
        precond = haznics.create_precond_amg(A_ptr, amgparam)

        # if fail, setup returns null
        if not precond:
            raise RuntimeError(
                "AMG levels failed to set up (null pointer returned) ")

        Precond.__init__(self, A, "AMG", parameters, precond)


class FAMG(Precond):
    """
    Fractional AMG preconditioner from the HAZmath Library

    """

    def __init__(self, A, M, parameters=None):
        # change data type for the matrices (to dCSRmat pointer)
        A_ptr = PETSc_to_dCSRmat(A)
        M_ptr = PETSc_to_dCSRmat(M)

        # initialize amg parameters (AMG_param pointer)
        amgparam = haznics.AMG_param()

        # set extra amg parameters
        parameters = parameters if (parameters and isinstance(parameters, dict)) \
            else {'fpwr': 0.5, 'smoother': 'fjacobi'}
        haznics.param_amg_set_dict(parameters, amgparam)

        # print (relevant) amg parameters
        haznics.param_amg_print(amgparam)

        # set FAMG preconditioner
        precond = haznics.create_precond_famg(A_ptr, M_ptr, amgparam)

        # if fail, setup returns null
        if not precond:
            raise RuntimeError(
                "FAMG levels failed to set up (null pointer returned) ")

        Precond.__init__(self, A, "FAMG", parameters, precond)


class RA(Precond):
    """
    Rational approximation preconditioner from the HAZmath library

    """

    def __init__(self, A, M, parameters=None):
        # change data type for the matrices (to dCSRmat pointer)
        A_ptr = PETSc_to_dCSRmat(A)
        M_ptr = PETSc_to_dCSRmat(M)

        # initialize amg parameters (AMG_param pointer)
        amgparam = haznics.AMG_param()

        # set extra amg parameters
        parameters = parameters if (parameters and isinstance(parameters, dict)) \
            else {'coefs': [1.0, 0.0], 'pwrs': [0.5, 0.0]}
        haznics.param_amg_set_dict(parameters, amgparam)

        # print (relevant) amg parameters
        haznics.param_amg_print(amgparam)

        # get scalings
        scaling_a = 1. / A.norm("linf")
        scaling_m = 1. / df.as_backend_type(M).mat().getDiagonal().min()[1]

        # get coefs and powers
        alpha, beta = parameters['coefs']
        s_power, t_power = parameters['pwrs']

        # set RA preconditioner #
        precond = haznics.create_precond_ra(A_ptr, M_ptr, s_power, t_power,
                                            alpha, beta, scaling_a, scaling_m,
                                            amgparam)

        # if fail, setup returns null
        if not precond:
            raise RuntimeError(
                "Rational Approximation data failed to set up (null pointer "
                "returned) ")

        Precond.__init__(self, A, "RA", parameters, precond)


class HXCurl(Precond):
    """
    HX preconditioner from the HAZmath library for the curl-curl inner product
    NB! only for 3D problems
    TODO: needs update and test
    """

    def __init__(self, Acurl, V, parameters=None):
        # get auxiliary operators
        mesh = V.mesh()
        Pc = Pcurl(mesh)
        Grad = discrete_gradient(mesh)

        # change data type for the matrices (to dCSRmat pointer)
        Acurl_ptr = PETSc_to_dCSRmat(Acurl)
        Pcurl_ptr = PETSc_to_dCSRmat(Pc)
        Grad_ptr = PETSc_to_dCSRmat(Grad)

        # initialize amg parameters (AMG_param pointer)
        amgparam = haznics.AMG_param()

        # set extra amg parameters
        if parameters and isinstance(parameters, dict):
            haznics.param_amg_set_dict(parameters, amgparam)
        else:
            parameters = {'prectype': haznics.PREC_HX_CURL_A}

        # print (relevant) amg parameters
        haznics.param_amg_print(amgparam)

        # add or multi
        try:
            prectype = parameters['prectype']
        except KeyError:
            prectype = haznics.PREC_HX_CURL_A

        # set HX CURL preconditioner (NB: this sets up both data and fct)
        precond = haznics.create_precond_hxcurl(Acurl_ptr, Pcurl_ptr, Grad_ptr,
                                                prectype, amgparam)

        # if fail, setup returns null
        if not precond:
            raise RuntimeError(
                "HXcurl data failed to set up (null pointer returned) ")
        """
        try:
            prectype = parameters['prectype']
        except KeyError:
            prectype = ''

        if prectype in ["add", "Add", "ADD", "additive", "ADDITIVE"]:
            precond.fct = haznics.precond_hx_curl_additive
        elif prectype in ["multi", "MULTI", "Multi", "multiplicative",
                          "MULTIPLICATIVE"]:
            precond.fct = haznics.precond_hx_curl_multiplicative
        else:  # default is additive
            precond.fct = haznics.precond_hx_curl_additive
        """
        Precond.__init__(self, Acurl, "HXCurl_add", parameters, precond)


class HXDiv(Precond):
    """
    HX preconditioner from the HAZmath library for the div-div inner product
    NB! Pdiv only works for 3D problems
    TODO: needs update and test
    """

    def __init__(self, Adiv, V, parameters=None):
        # get auxiliary operators
        mesh = V.mesh()
        Pd = Pdiv(mesh)
        Curl = discrete_curl(mesh)

        # change data type for the matrices (to dCSRmat pointer)
        Adiv_ptr = PETSc_to_dCSRmat(Adiv)
        Pdiv_ptr = PETSc_to_dCSRmat(Pd)
        Curl_ptr = PETSc_to_dCSRmat(Curl)

        # initialize amg parameters (AMG_param pointer)
        amgparam = haznics.AMG_param()

        # set extra amg parameters
        if parameters and isinstance(parameters, dict):
            haznics.param_amg_set_dict(parameters, amgparam)
        else:
            parameters = {'dim': mesh.topology().dim(),
                          'prectype': haznics.PREC_HX_DIV_A}

        # print (relevant) amg parameters
        haznics.param_amg_print(amgparam)

        # get dimension and type of HX precond application
        try:
            dim = parameters['dim']
        except KeyError:
            dim = 2

        # add or multi
        try:
            prectype = parameters['prectype']
        except KeyError:
            prectype = haznics.PREC_HX_DIV_A

        if dim == 3:
            Pc = Pcurl(mesh)

            # change data type for the Pcurl matrix (to dCSRmat pointer)
            Pcurl_ptr = PETSc_to_dCSRmat(Pc)

            # set HX DIV preconditioner (NB: this sets up both data and fct)
            precond = haznics.create_precond_hxdiv_3D(Adiv_ptr, Pdiv_ptr,
                                                      Curl_ptr, Pcurl_ptr,
                                                      prectype, amgparam)

            # if fail, setup returns null
            if not precond:
                raise RuntimeError(
                    "HXdiv data failed to set up (null pointer returned) ")
            """
            if prectype in ["add", "Add", "ADD", "additive", "ADDITIVE"]:
                precond.fct = haznics.precond_hx_div_additive

            elif prectype in ["multi", "MULTI", "Multi", "multiplicative", 
            "MULTIPLICATIVE"]:
                precond.fct = haznics.precond_hx_div_multiplicative

            else:
                # default is additive
                precond.fct = haznics.precond_hx_div_additive
            """
            Precond.__init__(self, Adiv, "HXDiv_add", parameters, precond)

        else:
            # set HX DIV preconditioner (NB: this sets up both data and fct)
            precond = haznics.create_precond_hxdiv_2D(Adiv_ptr, Pdiv_ptr,
                                                      Curl_ptr, prectype,
                                                      amgparam)

            # if fail, setup returns null
            if not precond:
                raise RuntimeError(
                    "HXdiv data failed to set up (null pointer returned) ")
            """
            if prectype in ["add", "Add", "ADD", "additive", "ADDITIVE"]:
                precond.fct = haznics.precond_hx_div_additive_2D

            elif prectype in ["multi", "MULTI", "Multi", "multiplicative", "MULTIPLICATIVE"]:
                precond.fct = haznics.precond_hx_div_multiplicative_2D

            else:
                # default is additive
                precond.fct = haznics.precond_hx_div_additive_2D
            """
            Precond.__init__(self, Adiv, "HXDiv_add", parameters, precond)


# ----------------------------------- EOF ----------------------------------- #
