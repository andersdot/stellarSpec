import matplotlib
matplotlib.use('pdf')
import numpy as np
from xdgmm import XDGMM
import drawEllipse
import matplotlib.pyplot as plt
import matplotlib as mpl
import testXD
import sys
import demo_plots as dp
import os
import stellarTwins as st
import scipy.integrate

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

    posteriorFile = 'posteriorParallax.' + str(ngauss) + 'gauss.dQ' + str(quantile) + '.' + iter + '.' + survey + '.' + dataFilename
    for file in [posteriorFile, 'posteriorSimple.npz']:
        data = np.load(posteriorFile)
        parallax = data['mean']
        parallax_err = np.sqrt(data['var'])
        c = np.log(data['var']) - np.log(tgas['parallax_error']**2.)
        absMagKinda = parallax*10.**(0.2*apparentMagnitude)
        absMagKinda_err = parallax_err*10.**(0.2*apparentMagnitude)
        y = absMagKinda
        yplus  = y + absMagKinda_err
        yminus = y - absMagKinda_err
        parallaxErrGoesNegative = yminus < 0
        absMagYMinus = testXD.absMagKinda2absMag(yminus)
        absMagYMinus[parallaxErrGoesNegative] = -50.
        yerr_minus = testXD.absMagKinda2absMag(y) - absMagYMinus
        yerr_plus = testXD.absMagKinda2absMag(yplus) - testXD.absMagKinda2absMag(y)
        dp.plot_sample(color, testXD.absMagKinda2absMag(y), sample[:,0], testXD.absMagKinda2absMag(sample[:,1]),
                    xdgmm, xerr=color_err, yerr=[yerr_minus, yerr_plus], xlabel=xlabel, ylabel=ylabel, xlim=xlim,
                    ylim=ylim, errSubsample=2.4e3, thresholdScatter=2., binsScatter=200, c=c)
    dataFile = 'inferredDistances_data_' + file.split('.')[0] + '.png'
    priorFile = 'prior_' + str(ngauss) +'gauss.png'
    os.rename('plot_sample.data.png', dataFile)
    os.rename('plot_sample.prior.png', priorFile)



def comparePosterior():
    ngauss = 128
    quantile = 0.05
    iter = '10th'
    survey = '2MASS'
    dataFilename = 'All.npz'
    posteriorFile = 'posteriorParallax.' + str(ngauss) + 'gauss.dQ' + str(quantile) + '.' + iter + '.' + survey + '.' + dataFilename
    data = np.load(posteriorFile)
    parallaxPost = data['posterior']
    parallaxMean = data['mean']
    parallaxVar = data['var']

    posteriorFile='posteriorSimple.npz'
    data = np.load(posteriorFile)
    parallaxSimplePost = data['posterior']
    parallaxSimpleMean = data['mean']
    parallaxSimpleVar = data['var']

    tgas, twoMass, Apass, bandDictionary, indices = testXD.dataArrays()

    fig, axes = plt.subplots(1,2)
    axes[0].plot(tgas['parallax_error']**2., parallaxVar, 'ko', markersize=1)
    axes[1].plot(tgas['parallax_error']**2., parallaxSimpleVar, 'ko', markersize=1)
    for ax in axes:
        ax.set_xlabel(r'$\sigma^2_{\varpi}$', fontsize=18)
        ax.set_ylabel(r'$\tilde{sigma}^2_{\varpi}$', fontsize=18)
    axes[0].set_title('CMD Prior')
    axes[1].set_title('Exp Dec Sp Den Prior')
    fig.savefig('varianceComparison.png')
    fig.close()

    fig, axes = plt.subplots(1, 2)
    axes[0].plot(tgas['parallax'], parallaxMean, 'ko', markersize=1)
    axes[1].plot(tgas['parallax'], parallaxSimpleMean, 'ko', markersize=1)
    for ax in axes:
        ax.set_xlabel(r'$\varpi$', fontsize=18)
        ax.set_ylabel(r'$E(\varpi)$', fontsize=18)
    axes[0].set_title('CMD Prior')
    axes[1].set_title('Exp Dec Sp Den Prior')

def compareCMD2Simple(ngauss=128, quantile=0.05, iter='10th', survey='2MASS', dataFilename='All.npz'):
    postFile = 'posteriorParallax.' + str(ngauss) + 'gauss.dQ' + str(quantile) + '.' + iter + '.' + survey + '.' + dataFilename
    data = np.load(postFile)
    mean = data['mean']
    var = data['var']
    posterior = data['posterior']

    dataSim = np.load('posteriorSimple.npz')
    meanSim = data['mean']
    varSim = data['var']
    posteriorSim = data['posterior']

    neg = tgas['parallax'] < 0
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(data['mean'][~neg], mean[~neg] - tgas['parallax'][~neg], 'ko', markersize=1)
    ax[0].plot(data['mean'][neg], mean[neg] - tgas['parallax'][neg], 'ro', markersize=1)
    ax[0].set_xscale('log')
    ax[1].plot(data['mean'][~neg], np.log(var[~neg]) - np.log(tgas['parallax_error'][~neg]**2.), 'ko', markersize=1)
    ax[1].plot(data['mean'][neg], np.log(var[neg]) - np.log(tgas['parallax_error'][neg]**2.), 'ro', markersize=1)
    ax[1].set_xscale('log')
    ax[0].set_xlabel(r'$E[\varpi]$', fontsize=18)
    ax[1].set_xlabel(r'$E[\varpi]$', fontsize=18)
    ax[0].set_ylabel(r'$E[\varpi] - \varpi$', fontsize=18)
    ax[1].set_ylabel(r'$ln \, \tilde{\sigma}_{\varpi}^2 - ln \, \sigma_{\varpi}^2$', fontsize=18)
    plt.tight_layout()

def examplePosterior(nexamples=100, postFile='posteriorSimple.npz', dustFile='dust.npz', nPosteriorPoints=1000, xdgmmFilename='xdgmm.fit'):
    tgas, twoMass, Apass, bandDictionary, indices = testXD.dataArrays()
    xdgmm = XDGMM(filename=xdgmmFilename)
    absmag = 'J'
    mag1 = 'J'
    mag2 = 'K'
    ndim = 2
    data = np.load(dustFile)
    dustEBV = data['ebv']
    absMagKinda, apparentMagnitude = testXD.absMagKindaArray(absmag, dustEBV, bandDictionary, tgas['parallax'])
    color = testXD.colorArray(mag1, mag2, dustEBV, bandDictionary)
    color_err = np.sqrt(bandDictionary[mag1]['array'][bandDictionary[mag1]['err_key']]**2. + bandDictionary[mag2]['array'][bandDictionary[mag2]['err_key']]**2.)
    xparallaxMAS = np.logspace(-2, 2, 1000)
    data = np.load(postFile)
    posterior = data['posterior']
    mean = data['mean']
    var = data['var']
    varDiff = var - tgas['parallax_error']**2.
    ind = np.argsort(varDiff)[::-1]
    for i in ind[0:nexamples]:
        xabsMagKinda = testXD.parallax2absMagKinda(xparallaxMAS, apparentMagnitude[i])
        likelihood = st.gaussian(tgas['parallax'][i], tgas['parallax_error'][i], xparallaxMAS)
        meanPrior, covPrior = testXD.matrixize(color[i], absMagKinda[i], color_err[i], 1e3)
        meanPrior = meanPrior[0]
        covPrior = covPrior[0]
        allMeans, allAmps, allCovs, summedPriorAbsMagKinda = testXD.absMagKindaPosterior(xdgmm, ndim, meanPrior, covPrior, xabsMagKinda, projectedDimension=1, nPosteriorPoints=nPosteriorPoints, prior=True)
        norm = scipy.integrate.cumtrapz(summedPriorAbsMagKinda*10.**(0.2*apparentMagnitude[i]), x=xparallaxMAS)[-1]
        plotPrior = summedPriorAbsMagKinda*10.**(0.2*apparentMagnitude[i])/norm
        posteriorFly = likelihood*summedPriorAbsMagKinda*10.**(0.2*apparentMagnitude[i])
        norm = scipy.integrate.cumtrapz(posteriorFly, x=xparallaxMAS)[-1]
        posteriorFly = posteriorFly/norm
        plt.clf()
        plt.plot(xparallaxMAS, posterior[i], label='posterior')
        plt.plot(xparallaxMAS, likelihood, label='likelhood')
        plt.plot(xparallaxMAS, plotPrior, label='prior')
        plt.plot(xparallaxMAS, posteriorFly, label='posterior on the Fly')
        plt.xlim(tgas['parallax'][i] - 5.*tgas['parallax_error'][i], tgas['parallax'][i] + 5.*tgas['parallax_error'][i])
        #plt.xscale('log')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.xlabel('parallax [mas]', fontsize=18)
        plt.title('J-K: ' + '{0:.1f}'.format(color[i]) + '    M: ' +  '{0:.1f}'.format(testXD.absMagKinda2absMag(absMagKinda[i])))
        plt.savefig('exampleCMDPosteriorLargerVariance_' + str(i) + '.png')


def compareSimpleGaia(ngauss=128, quantile=0.05, iter='10th', survey='2MASS', dataFilename='All.npz'):
    postFile = 'posteriorParallax.' + str(ngauss) + 'gauss.dQ' + str(quantile) + '.' + iter + '.' + survey + '.' + dataFilename
    yim = (-1, 5)
    for file in ['posteriorSimple.npz', postFile]:
        data = np.load(file)
        mean = data['mean']
        var = data['var']
        tgas, twoMass, Apass, bandDictionary, indices = testXD.dataArrays()

        neg = tgas['parallax'] < 0
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(data['mean'][~neg], mean[~neg] - tgas['parallax'][~neg], 'ko', markersize=0.5)
        ax[0].plot(data['mean'][neg], mean[neg] - tgas['parallax'][neg], 'ro', markersize=0.5)
        ax[0].set_xscale('log')
        ax[1].plot(data['mean'][~neg], np.log(var[~neg]) - np.log(tgas['parallax_error'][~neg]**2.), 'ko', markersize=0.5)
        ax[1].plot(data['mean'][neg], np.log(var[neg]) - np.log(tgas['parallax_error'][neg]**2.), 'ro', markersize=0.5)
        ax[1].set_xscale('log')
        ax[0].set_xlabel(r'$E[\varpi]$', fontsize=18)
        ax[1].set_xlabel(r'$E[\varpi]$', fontsize=18)
        ax[0].set_ylabel(r'$E[\varpi] - \varpi$', fontsize=18)
        ax[1].set_ylabel(r'$\mathrm{ln} \, \tilde{\sigma}_{\varpi}^2 - \mathrm{ln} \, \sigma_{\varpi}^2$', fontsize=18)
        plt.tight_layout()
        #if file == 'posteriorSimple.npz':
        ax[0].set_ylim(-5, 5)
        ax[1].set_ylim(-6, 2)
        ax[0].set_xlim(1e-1, 1e1)
        ax[1].set_xlim(1e-1, 1e2)
        fig.savefig(file.split('.')[0] + '_Comparison2Gaia.png')
        notnans = ~np.isnan(var) & ~np.isnan(tgas['parallax_error'])
        print np.median(np.log(var[notnans]) - np.log(tgas['parallax_error'][notnans]**2.))

if __name__ == '__main__':
    #comparePrior()
    quantile = np.float(sys.argv[1])
    ngauss = np.int(sys.argv[2])
    if ngauss == 128: iter='10th'
    if ngauss == 512: iter='4th'
    if ngauss == 2048: iter='1st'
    Nsamples=1.2e5
    survey='2MASS'
    dataFilename = 'All.npz'
    xdgmmFilename = 'xdgmm.'             + str(ngauss) + 'gauss.dQ' + str(quantile) + '.' + iter + '.' + survey + '.' + dataFilename + '.fit'
    postFile = 'posteriorParallax.' + str(ngauss) + 'gauss.dQ' + str(quantile) + '.' + iter + '.' + survey + '.' + dataFilename
    dustFile      = 'dustCorrection.'    + str(ngauss) + 'gauss.dQ' + str(quantile) + '.' + iter + '.' + survey + '.' + dataFilename
    #dustViz(quantile=quantile)
    #
    examplePosterior(postFile=postFile, nexamples=20, dustFile=dustFile, xdgmmFilename=xdgmmFilename)
    dataViz(ngauss=ngauss, quantile=quantile, iter=iter, Nsamples=Nsamples)
    #compareSimpleGaia()
