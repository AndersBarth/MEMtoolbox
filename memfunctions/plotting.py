from memfunctions.mem import A2pdf
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
from matplotlib.pylab import figure
from numpy import amax, cumsum


# define plotting function
def plot_noreg(x, data, F_R, Rrange, mR, Rfract_noreg, chi2_opt):
    # Plot Optimization results
    gs = grd.GridSpec(1, 2, figure=figure(figsize=(10, 5)))

    axD = plt.subplot(gs[0, 0])
    axD.plot(x, data, 'r.')
    axD.plot(x, F_R.dot(Rfract_noreg), 'b')
    axD.set_xlabel('R [nm]', fontsize=15)
    axD.set_ylabel('Counts', fontsize=15)
    axD.set_xlim(min(x), 40)
    axD.set_ylim(0, 1.05*max(data))
    axD.set_title('$\chi^2_{No \, opt}$ = %6.3f' % (chi2_opt), fontsize=15)

    axP = plt.subplot(gs[0, 1])
    axP.plot(Rrange,  A2pdf(Rfract_noreg, mR, Rrange), 'b')
    axP.set_xlabel('R [nm]', fontsize=15)
    axP.set_ylabel('P(R)', fontsize=15)
    axP.set_xlim(-1, amax(Rrange))
    axP.set_ylim(0,)
    axP.set_title('sum(P) = %f' % sum(Rfract_noreg))


def plot_Tik(selection_i, x, data, Rrange, mR, pTdist, F_R, chiTrange,
             muTrange, nTrange):
    gs = grd.GridSpec(3, 2, figure=figure(figsize=(15, 15)))
    axD = plt.subplot(gs[2, 0])
    axD.plot(x, data, 'r')
    axD.plot(x, F_R.dot(pTdist[selection_i]), 'b')
    axD.set_xlabel('R [nm]', fontsize=15)
    axD.set_ylabel('Counts', fontsize=15)
    axD.set_xlim(min(x), 40)
    axD.set_ylim(0., 1.05*max(data))

    axM = plt.subplot(gs[0, 0])
    axM.semilogy(chiTrange, muTrange, 'b')
    axM.plot(chiTrange[selection_i],
             muTrange[selection_i], 'bo', markersize=10)
    #axM.axvline(chi2_est, color='r')
    axM.axhline(muTrange[selection_i], color='b')
    axM.axvline(chiTrange[selection_i], color='b')
    axM.set_xlabel(r'$\chi^2$', fontsize=15)
    axM.set_ylabel(r'$\mu$', fontsize=15)
    axM.set_xlim(min(chiTrange), max(chiTrange))
    axM.set_ylim(min(muTrange), max(muTrange))
    axM.set_title(r'$\chi^2_{Tikh} = %6.3f, \ \ \mu = %6.3f$' % (
        chiTrange[selection_i], muTrange[selection_i]), fontsize=15)

    axL = plt.subplot(gs[1, 0])
    axL.plot(chiTrange, nTrange, 'b')
    axL.plot(chiTrange[selection_i], nTrange[selection_i], 'bo', markersize=10)
    #axL.axvline(chi2_AV_est, color='r')
    axL.axvline(chiTrange[selection_i], color='b')
    axL.set_xlim(min(chiTrange), max(chiTrange))
    axL.set_ylim(0, max(nTrange))
    axL.set_xlabel(r'$\chi^2$', fontsize=15)
    axL.set_ylabel('norm(p)', fontsize=15)

    axPm = plt.subplot(gs[0, 1])
    axPm.imshow(pTdist, cmap='Blues', aspect='auto', origin='upper')
    axPm.axhline(selection_i, color='blue')
    axPm.set_xlabel('$R$ index', fontsize=15)
    axPm.set_ylabel('$\mu$ index', fontsize=15)

    axP = plt.subplot(gs[1, 1])
    # axP.plot(Rrange,  A2pdf(Rfract_noreg,mR,Rrange)/10.0, 'dodgerblue');
    # axP.plot(RrangeAV,A2pdf(RfractAV,mR_AV,RrangeAV), 'r');
    axP.plot(Rrange,  A2pdf(pTdist[selection_i], mR, Rrange), 'b')
    axP.set_xlim(Rrange[0], Rrange[-1])
    axP.set_ylim(0,)
    axP.set_xlabel('R', fontsize=15)
    axP.set_ylabel('pdf (R)', fontsize=15)

    axC = plt.subplot(gs[2, 1])
    # axC.plot(Rrange, cumsum(Rfract_noreg/sum(Rfract_noreg)), 'dodgerblue');
    #axC.plot(RrangeAV,cumsum(RfractAV), 'r');
    axC.plot(Rrange, cumsum(pTdist[selection_i]/sum(pTdist[selection_i])), 'b')
    axC.set_xlim(Rrange[0], Rrange[-1])
    axP.set_ylim(0,)
    axC.set_ylim(0, 1)
    axC.set_xlabel('R', fontsize=15)
    axC.set_ylabel('cdf (R)', fontsize=15)


def plot_MEM(selection_i, x, data, Rrange, mR, pMdist, F_R, chiMrange,
             muMrange, Srange):
    gs = grd.GridSpec(3, 2, figure=figure(figsize=(15, 15)))

    axD = plt.subplot(gs[2, 0])
    axD.plot(x, data, 'r')
    axD.plot(x, F_R.dot(pMdist[selection_i]), 'b')
    axD.set_xlabel('x', fontsize=15)
    axD.set_ylabel('y', fontsize=15)
    axD.set_xlim(x[0], x[-1])
    axD.set_ylim(0, 1.05*max(data))

    axM = plt.subplot(gs[0, 0])
    # axM.axvline(chi2_AV_est, color='r')
    axM.semilogy(chiMrange, muMrange, 'b')
    axM.plot(chiMrange[selection_i],
             muMrange[selection_i], 'bo', markersize=10)
    axM.axhline(muMrange[selection_i], color='b')
    axM.axvline(chiMrange[selection_i], color='b')
    axM.set_xlabel(r'$\chi^2$', fontsize=15)
    axM.set_ylabel(r'$\mu$', fontsize=15)
    axM.set_xlim(min(chiMrange), max(chiMrange))
    axM.set_ylim(min(muMrange), max(muMrange))
    axM.set_title(r'$\chi^2_{ME} = %6.3f, \ \ \mu = %6.3f$' % (
        chiMrange[selection_i], muMrange[selection_i]), fontsize=15)

    axL = plt.subplot(gs[1, 0])
    # axL.axvline(chi2_AV_est, color='r')
    axL.plot(chiMrange, -Srange, 'b')
    axL.plot(chiMrange[selection_i], -Srange[selection_i], 'bo', markersize=10)
    axL.axvline(chiMrange[selection_i], color='b')
    axL.set_xlim(min(chiMrange), max(chiMrange))
    axL.set_ylim(min(-Srange), max(-Srange))
    axL.set_xlabel(r'$\chi^2$', fontsize=15)
    axL.set_ylabel('-S(p,m)', fontsize=15)

    axPm = plt.subplot(gs[0, 1])
    axPm.imshow(pMdist, cmap='Blues', aspect='auto', origin='upper',
                extent=(Rrange[0], Rrange[-1], len(Rrange), 0))
    axPm.axhline(selection_i, color='blue')
    axPm.set_xlabel('Parameter', fontsize=15)
    axPm.set_ylabel('$\mu$ index', fontsize=15)

    axP = plt.subplot(gs[1, 1])
    # axP.plot(Rrange,  A2pdf(Rfract_noreg,mR,Rrange)/10.0, 'dodgerblue');
    # axP.plot(RrangeAV,A2pdf(RfractAV,mR_AV,RrangeAV), 'r');
    axP.plot(Rrange,  A2pdf(pMdist[selection_i], mR, Rrange), 'b')
    axP.set_xlim(Rrange[0], Rrange[-1])
    axP.set_ylim(0,)
    axP.set_xlabel('R', fontsize=15)
    axP.set_ylabel('pdf (R)', fontsize=15)

    axC = plt.subplot(gs[2, 1])
    # axC.plot(RrangeAV,cumsum(RfractAV), 'r');
    axC.plot(Rrange, cumsum(pMdist[selection_i]/sum(pMdist[selection_i])), 'b')
    axC.set_xlim(Rrange[0], Rrange[-1])
    axP.set_ylim(0,)
    axC.set_ylim(0, 1)
    axC.set_xlabel('R', fontsize=15)
    axC.set_ylabel('cdf (R)', fontsize=15)
