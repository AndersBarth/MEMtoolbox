import scipy.stats
import numpy as np


def norm_pdf(x, m=0.0, s=1.0):
    return scipy.stats.norm.pdf(x, m, s)


def exp_pdf(x, tau=1):
    return scipy.stats.expon.pdf(x, tau)
