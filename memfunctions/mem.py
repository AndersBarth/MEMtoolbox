from numpy import log, log10, linalg, gradient, logspace, zeros, array, \
    identity, arange, linspace, asarray, diag, sqrt, ndarray, sum
from cvxopt import matrix, solvers
from numba import jit

solvers.options['show_progress'] = False
solvers.options['maxiters'] = 200


# convert amplitude to probabily density function
def A2pdf(ax, mx, xrange):
    p = ax/sum(ax)
    Tx = xrange[-1]-xrange[0]
    return p*mx*sum(1.0/mx)/Tx


def chi2comp(fexp, sd, F):
    Nsel = fexp.size
    s = 1.0/sd
    Fn = F*(s[:, None])
    fn = fexp*s
    c = fn.dot(fn)/Nsel   # scalar
    q = -2.0*Fn.T.dot(fn)/Nsel   # vector
    H = 2.0*Fn.T.dot(Fn)/Nsel   # matix
    return c, q, H


def densityp(xrange):
    dens = 1.0/gradient(xrange, edge_order=2)
    return dens/sum(dens)


def getchi2(p, c, q, H):
    return c + (q + 0.5*p.T.dot(H)).dot(p)


def medelta(p, l, q, H):
    Q = q + H.dot(p)
    if any(l):
        return 0.5*linalg.norm(Q/linalg.norm(Q)+(l+1.0)/linalg.norm(l+1.0))
    else:
        return 0.5*linalg.norm(Q/linalg.norm(Q))


# calculate the kernel
def kernel_calculation(x, data, kernel_function, param_range):
    mR = densityp(param_range)

    F_R = ndarray(shape=(len(x), len(param_range)))
    # Model Kernel calculation
    for (i, R) in enumerate(param_range):
        F_R[:, i] = kernel_function(R)

    # Calculate chi2 decomposition:
    v = data
    sd = sqrt(v)
    sd[sd == 0] = 1
    c_R, q_R, H_R = chi2comp(data,
                             sd,
                             F_R)
    return F_R, param_range, mR, c_R, q_R, H_R


# MEM regularization
def solve_MEM(c_R, q_R, H_R, mR, mu_min=0.001, mu_max=1, n_mu=100, niter=50,
              sel_i=0):
    muMrange = logspace(log10(mu_min), log10(mu_max), n_mu)
    if sel_i != 0:
        muMrange = array([muMrange[sel_i]])

    Srange = zeros(muMrange.size)
    chiMrange = zeros(muMrange.size)
    pMdist = zeros([muMrange.size, q_R.size])
    dMrange = zeros([muMrange.size, niter])

    for i in arange(muMrange.size):
        mu = muMrange[i]
        j = 0

        pm = mR.copy()
        D = diag(zeros(pm.size))
        l = zeros(pm.size)
        d = medelta(pm, l, q_R, H_R)
        dMrange[i, j] = d

        while (d > 0.01) and j < niter:

            sol_me = solvers.qp(matrix(H_R + 2.0*mu*D),
                                matrix(q_R + mu*(l-1.0)),
                                matrix(-identity(q_R.size), tc='d'),
                                matrix(zeros(q_R.size), tc='d')
                                )
            pm = asarray(sol_me['x'])[:, 0]
            pm[pm < 1.0e-15] = 1.0e-15
            l = log(pm/mR)
            D = diag(1.0/pm)
            d = medelta(pm, l, q_R, H_R)
            dMrange[i, j] = d
            j += 1

        pMdist[i] = pm
        chiMrange[i] = getchi2(pm, c_R, q_R, H_R)
        Srange[i] = -pm.dot(l)

    return pMdist, muMrange, chiMrange, Srange


# no regularization
def solve_noreg(c_R, q_R, H_R):
    sol_noreg = solvers.qp(matrix(H_R),
                           matrix(q_R),
                           matrix(-identity(q_R.size), tc='d'),
                           matrix(zeros(q_R.size), tc='d'))
    Rfract_noreg = asarray(sol_noreg['x'])[:, 0]

    return Rfract_noreg, getchi2(Rfract_noreg, c_R, q_R, H_R)


# Tikhonov regularization
def solve_Tik(c_R, q_R, H_R, mu_min=0.01, mu_max=100, n_mu=100):
    # Scan regularization levels mu
    muTrange = logspace(log10(mu_min), log10(mu_max), n_mu)
    nTrange = zeros(muTrange.size)
    chiTrange = zeros(muTrange.size)
    pTdist = zeros([muTrange.size, q_R.size])
    for i in arange(muTrange.size):
        mu = muTrange[i]
        # Modified H for Tikhonov regularization
        sol_tikh = solvers.qp(matrix(H_R+2.0*mu*identity(q_R.size)),
                              matrix(q_R),
                              matrix(-identity(q_R.size), tc='d'),
                              matrix(zeros(q_R.size), tc='d'))
        pTikh = asarray(sol_tikh['x'])[:, 0]
        nTrange[i] = linalg.norm(pTikh)
        chiTrange[i] = getchi2(pTikh, c_R, q_R, H_R)
        pTdist[i] = pTikh

    return pTdist, muTrange, chiTrange, nTrange
