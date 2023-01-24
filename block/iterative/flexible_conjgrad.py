from .common import inner
from math import sqrt

def precondconjgrad(B, A, x, b, tolerance, maxiter, progress, m_max=1,
                    relativeconv=False, robustresidual=None, callback=None):
    #####
    # Implemented from algorithm 2.1 -- FCG(m_max) in "Flexible Conjugate
    # Gradients" <https://doi.org/10.1137/S1064827599362314>, Y. Notay (2000)
    #
    # Explicit Gram-Schmidt orthogonalization with up to the previous m_max
    # search directions is performed, partially restarted every m_max+1
    # iterations. The purpose is to preserve local orthogonality even with
    # inexact (nonlinear/variable) preconditioning.
    #
    # If B is fixed, m_max=1 should have the same convergence as regular CG, at
    # the cost of one extra inner product per iteration. If B is variable
    # between iterations, it should have more stable convergence. Using larger
    # values for m_max primarily pays off when the preconditioner varies *and*
    # there is potential for superlinear convergence despite (small) isolated
    # eigenvalues -- ref tables 2 and 3 in Notay (2000).
    #
    # m_max=1, i.e. orthogonality with only the previous search direction, is
    # claimed by Notay to be equivalent to the Polak-Ribiere formulation for
    # beta used in Golub and Ye (2000) as well as the Wikipedia article. This
    # version is also found to be robust wrt nonsymmetric preconditioning (GMG
    # without post-smoothing) in Bouwmeester et al (2015).
    #####

    r = b - A*x

    iter = 0
    alphas = []
    betas = []
    residuals = []
    prev = []

    while True:
        #
        # Calculate iteration _i_
        #

        Br = B*r
        residual = sqrt(abs(inner(r,Br)))
        residuals.append(residual)

        if callable(callback):
            if callback(k=iter, x=x, r=residual):
                break
        if relativeconv:
            if residuals and residual/residuals[0] <= tolerance:
                break
        else:
            if residual <= tolerance:
                break

        Br_prev_Ad = [inner(Br, Ad_) for _, Ad_, _ in prev]
        d = Br; del Br
        for BrAd_, (d_, _, dAd_) in zip(Br_prev_Ad, prev):
            #d -= BrAd_/dAd_ * d_
            d.axpy(-BrAd_/dAd_, d_)

        Ad = A*d
        dAd = inner(d,Ad)
        if dAd == 0:
            print(f'ConjGrad breakdown')
            break
        alpha = inner(r,d)/dAd

        #
        # Prepare iteration _i+1_
        #
        iter += 1
        if iter > maxiter:
            break
        progress += 1

        if m_max > 0:
            if iter%(m_max+1) <= 1:
                prev.clear()
            prev.append((d, Ad, dAd))

        #x += alpha * d
        x.axpy(alpha, d)
        if robustresidual or (robustresidual is None and iter%50 == 0):
            r = b - A*x
        else:
            #r -= alpha * Ad
            r.axpy(-alpha, Ad)

    return x, residuals, alphas, betas
