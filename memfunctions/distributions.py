import scipy.stats
import numpy as np
from scipy.special import i0


def norm_pdf(x, m=0.0, s=1.0):
    return scipy.stats.norm.pdf(x, m, s)


def exp_pdf(x, tau=1):
    return scipy.stats.expon.pdf(x, tau)


def chi_distribution(x, loc=0.0, scale=1.0, norm=True):
    """Probability density function of a non-central chi-distribution in two dimensions.
    :param x: sampled distances
    :param loc: mean (location)
    :param scale: sigma parameter
    :param norm: Boolean if true the returned array is normalized to one
    :return: probability density
    """
    p = (x / (scale**2)) * np.exp(-(x**2+loc**2)/(2*scale**2)) * i0(x*loc/(scale**2))

    if norm:
        p /= p.sum()
    return p
