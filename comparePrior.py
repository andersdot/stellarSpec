import numpy as np
from xdgmm import XDGMM
import drawEllipse
import matplotlib.pyplot as plt
import matplotlib as mpl
import testXD
import sys
import demo_plots as dp
import os

def prior(xdgmm, ax):
    for gg in range(xdgmm.n_components):
        points = drawEllipse.plotvector(xdgmm.mu[gg], xdgmm.V[gg])
        ax[0].plot(points[0,:],testXD.absMagKinda2absMag(points[1,:]), c, lw=1, alpha=xdgmm.weights[gg]/np.max(xdgmm.weights))


def dustFilename(ngauss, quantile, iter, survey, dataFilename):
    return 'dustCorrection.'    + str(ngauss) + 'gauss.dQ' + str(quantile) + '.' + iter + '.' + survey + '.' + dataFilename


def comparePrior():
    ngauss = [512, 128]
    iter = ['1st', '6th']
    color = ['k', 'red']
    label = ['512 Gaussians', '128 Gaussians']
    fig, ax = plt.subplots(1,2, figsize=(12,5))

    for n, i, c, l in zip(ngauss, iter, color, label):
        xdgmmFilename = 'xdgmm.' + str(n) + 'gauss.dQ0.05.' + i + '.2MASS.All.npz.fit'
        xdgmm = XDGMM(filename=xdgmmFilename)

        for gg in range(xdgmm.n_components):
            if xdgmm.weights[gg] == np.max(xdgmm.weights):
                lab = l
            else:
                lab = None
            points = drawEllipse.plotvector(xdgmm.mu[gg], xdgmm.V[gg])
            ax[0].plot(points[0,:],testXD.absMagKinda2absMag(points[1,:]), c, lw=1, alpha=xdgmm.weights[gg]/np.max(xdgmm.weights))
            ax[1].plot(points[0,:], points[1,:], c, lw=1, alpha=xdgmm.weights[gg]/np.max(xdgmm.weights), label=lab)

    for a in ax:
        a.set_xlim(-0.5, 1.5)
        a.set_xlabel(r'$(J - K)^C$')
    ax[0].set_ylabel(r'$M_J^C$')
    ax[1].set_ylabel(r'$\varpi 10^{0.2\,m_J}$')
    ax[0].set_ylim(6, -6)
    ax[1].set_ylim(1100, -100)
    ax[1].legend(loc='lower left', fontsize=10)
    plt.tight_layout()
    fig.savefig('priorNgaussComparison.png')

def dustViz(ngauss=128, quantile=0.5, iter='8th', survey='2MASS', dataFilename='All.npz'):

    tgas, twoMass, Apass, bandDictionary, indices = testXD.dataArrays()
    dustFile = dustFilename(ngauss, quantile, iter, survey, dataFilename)
    data = np.load(dustFile)
    dust = data['ebv']
    fig, ax = plt.subplots(figsize=(12,7))
    norm = mpl.colors.PowerNorm(gamma=1/2.)#(vmin=-0.5, vmax=1)
    im = ax.scatter(tgas['l'], tgas['b'], c=dust, lw=0, cmap='Greys', s=1, vmin=0, vmax=1, norm=norm)
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    ax.set_xlabel('$\mathscr{l}$ [deg]')
    ax.set_ylabel('$\mathscr{b}$ [deg]')
    cb = plt.colorbar(im, ax=ax)
    cb.set_clim(-0.1, 1)
    cb.set_label(r'E($B-V$ )')
    fig.savefig('dustViz.dQ' + str(quantile) + '.png')

def dataViz(survey='2MASS', ngauss=128, quantile=0.05, dataFilename='All.npz', iter='10th', Nsamples=3e5):

    if survey == 'APASS':
        mag1 = 'B'
        mag2 = 'V'
        absmag = 'G'
        xlabel='B-V'
        ylabel = r'M$_\mathrm{G}$'
        xlim = [-0.2, 2]
        ylim = [9, -2]

    if survey == '2MASS':
        mag1 = 'J'
        mag2 = 'K'
        absmag = 'J'
        xlabel = 'J-K$_s$'
        ylabel = r'M$_\mathrm{J}$'
        xlim = [-0.25, 1.25]
        ylim = [6, -6]

    xdgmmFilename = 'xdgmm.'             + str(ngauss) + 'gauss.dQ' + str(quantile) + '.' + iter + '.' + survey + '.' + dataFilename + '.fit'

    tgas, twoMass, Apass, bandDictionary, indices = testXD.dataArrays()
    dustEBV = 0.0
    color = testXD.colorArray(mag1, mag2, dustEBV, bandDictionary)
    absMagKinda, apparentMagnitude = testXD.absMagKindaArray(absmag, dustEBV, bandDictionary, tgas['parallax'])

    color_err = np.sqrt(bandDictionary[mag1]['array'][bandDictionary[mag1]['err_key']]**2. + bandDictionary[mag2]['array'][bandDictionary[mag2]['err_key']]**2.)
    absMagKinda_err = tgas['parallax_error']*10.**(0.2*bandDictionary[absmag]['array'][bandDictionary[absmag]['key']])

    xdgmm = XDGMM(filename=xdgmmFilename)
    sample = xdgmm.sample(Nsamples)
    negParallax = sample[:,1] < 0
    nNegP = np.sum(negParallax)
    while nNegP > 0:
        sampleNew = xdgmm.sample(nNegP)
        sample[negParallax] = sampleNew
        negParallax = sample[:,1] < 0
        nNegP = np.sum(negParallax)
    positive = absMagKinda > 0
    y = absMagKinda[positive]
    yplus  = y + absMagKinda_err[positive]
    yminus = y - absMagKinda_err[positive]
    parallaxErrGoesNegative = yminus < 0
    absMagYMinus = testXD.absMagKinda2absMag(yminus)
    absMagYMinus[parallaxErrGoesNegative] = -50.
    yerr_minus = testXD.absMagKinda2absMag(y) - absMagYMinus
    yerr_plus = testXD.absMagKinda2absMag(yplus) - testXD.absMagKinda2absMag(y)
    #yerr_minus = testXD.absMagKinda2absMag(yplus) - testXD.absMagKinda2absMag(y)
    #yerr_plus = testXD.absMagKinda2absMag(y) - absMagYMinus
    """
    testfig, testax = plt.subplots(3)
    testax[0].scatter(testXD.absMagKinda2absMag(y), y, s=1)
    testax[0].set_xlabel('absMag')
    testax[0].set_ylabel('absMagKinda')
    testax[1].scatter(testXD.absMagKinda2absMag(y), absMagYMinus, s=1)
    testax[1].set_xlabel('absMag')
    testax[1].set_ylabel('absMag Minus')
    testax[2].scatter(testXD.absMagKinda2absMag(y), testXD.absMagKinda2absMag(yplus), s=1)
    testax[2].set_xlabel('absMag')
    testax[2].set_ylabel('absMag Plus')
    plt.show()
    """
    dp.plot_sample(color[positive], testXD.absMagKinda2absMag(y), sample[:,0], testXD.absMagKinda2absMag(sample[:,1]),
                xdgmm, xerr=color_err[positive], yerr=[yerr_minus, yerr_plus], xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, errSubsample=2.4e3, thresholdScatter=2., binsScatter=200)
    dataFile = 'data_noDust.png'
    priorFile = 'prior_' + str(ngauss) +'gauss.png'
    os.rename('plot_sample.data.png', dataFile)
    os.rename('plot_sample.prior.png', priorFile)
    #import pdb; pdb.set_trace()

if __name__ == '__main__':
    #comparePrior()
    quantile = np.float(sys.argv[1])
    ngauss = np.int(sys.argv[2])
    if ngauss == 128: iter='10th'
    if ngauss == 512: iter='4th'
    if ngauss == 2048: iter='1st'
    Nsamples=1.2e5
    #dustViz(quantile=quantile)
    dataViz(ngauss=ngauss, quantile=quantile, iter=iter, Nsamples=Nsamples)
