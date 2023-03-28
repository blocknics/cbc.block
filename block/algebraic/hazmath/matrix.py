from block.algebraic.hazmath.precond import PETSc_to_dCSRmat, Precond
from block import block_mat, supports_mpi
import haznics
import dolfin as df


class RA(Precond):
    """
    Rational approximation from the HAZmath library

    """

    def __init__(self, A, M, parameters=None):
        supports_mpi(False, 'HAZMath does not work in parallel')

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
        precond = haznics.create_ra(A_ptr, M_ptr, s_power, t_power,
                                    alpha, beta, scaling_a, scaling_m,
                                    ra_tol=parameters.get('AAA_tol', 1E-10),
                                    amgparam=amgparam)

        # if fail, setup returns null
        if not precond:
            raise RuntimeError(
                "Rational Approximation data failed to set up (null pointer "
                "returned) ")

        Precond.__init__(self, A, "RA", parameters, amgparam, precond)
