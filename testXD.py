import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import os
import sys
import stellarTwins as st
from extreme_deconvolution import extreme_deconvolution as ed
import astropy.units as units
import numpy as np
import pdb
import time
from sklearn.model_selection import train_test_split
from astropy.io import fits
from xdgmm import XDGMM
from sklearn.learning_curve import validation_curve
from sklearn.cross_validation import ShuffleSplit
import demo_plots as dp
import drawEllipse
from astropy.coordinates import SkyCoord
import astropy.units as units
import scipy.integrate
import time
from matplotlib.colors import LogNorm
from dustmaps.bayestar import BayestarQuery

def convert2gal(ra, dec):
    """
    convert ra and dec to l and b, sky coordinates to galactic coordinates
    """
    return SkyCoord([ra, dec], unit=(units.hourangle, units.deg))

def m67indices(tgas, plot=False, dl=0.1, db=0.1, l='215.6960', b='31.8963'):
    """
    return the indices of tgas which fall within dl and db of l and b.
    default is M67
    """

    index = (tgas['b'] < np.float(b) + db) & \
            (tgas['b'] > np.float(b) - db) & \
            (tgas['l'] < np.float(l) + dl) & \
            (tgas['l'] > np.float(l) - dl)
    if plot:
        fig, ax = plt.subplots()
        ax.scatter(tgas['l'][index], tgas['b'][index], alpha=0.5, lw=0)
        fig.savefig('clusterOnSky.png')
    return index

def absMagKinda2Parallax(absMagKinda, apparentMag):
    """
    convert my funny units of parallax[mas]*10**(0.2*apparent magnitude[mag]) to parallax[mas]
    """
    return absMagKinda/10.**(0.2*apparentMag)


def parallax2absMagKinda(parallaxMAS, apparentMag):
    """
    convert parallax to my funny units of parallax[mas]*10**(0.2*apparent magnitude[mag])
    """
    return parallaxMAS*10.**(0.2*apparentMag)


def absMagKinda2absMag(absMagKinda):
    """
    convert my funny units of parallax[mas]*10**(0.2*apparent magnitude[mag]) to an absolute magnitude [mag]
    """
    absMagKinda_in_arcseconds = absMagKinda/1e3 #first convert parallax from mas ==> arcseconds
    return 5.*np.log10(10.*absMagKinda_in_arcseconds)


def XD(data_1, err_1, data_2, err_2, ngauss=2, mean_guess=np.array([[0.5, 6.], [1., 4.]]), w=0.0):
    """
    run XD
    """
    amp_guess = np.zeros(ngauss) + 1.
    ndim = 2
    X, Xerr = matrixize(data1, data2, err1, err2)
    cov_guess = np.zeros(((ngauss,) + X.shape[-1:] + X.shape[-1:]))
    cov_guess[:,diag,diag] = 1.0
    ed(X, Xerr, amp_guess, mean_guess, cov_guess, w=w)
    return amp_guess, mean_guess, cov_guess

def scalability(X, Xerr, numberOfStars = [1024, 2048, 4096, 8192]):
    """
    test the scalability of XD with various numbers of stars
    """
    totTime = np.zeros(4)
    for i, ns in enumerate(numberOfStars):
        totalTime, numStar = timing(X, Xerr, nstars=ns, ngauss=64)
        print totalTime, numStar
        totTime[i] = totalTime
        plt.plot(numData, totTime)
        plt.savefig('timing64Gaussians.png')


def timing(X, Xerr, nstars=1024, ngaussians=64):
    """
    test how long it takes for XD to run with nstars and ngaussians
    """
    amp_guess = np.zeros(ngaussians) + np.random.rand(ngaussians)
    cov_guess = np.zeros(((ngaussians,) + X.shape[-1:] + X.shape[-1:]))
    cov_guess[:,diag,diag] = 1.0
    mean_guess = np.random.rand(ngauss,2)*10.
    start = time.time()
    ed(X, Xerr, amp_guess, mean_guess, cov_guess)
    end = time.time()
    return end-start, nstars

def subset(data1, data2, err1, err2, nsamples=1024):
    """
    return a random subset of the data with nsamples
    """
    ind = np.random.randint(0, len(data1), size=nsamples)
    return matrixize(data1[ind], data2[ind], err1[ind], err2[ind])

def matrixize(data1, data2, err1, err2):
    """
    vectorize the 2 pieces of data into a 2D mean and 2D covariance matrix
    """
    X = np.vstack([data1, data2]).T
    Xerr = np.zeros(X.shape + X.shape[-1:])
    diag = np.arange(X.shape[-1])
    Xerr[:, diag, diag] = np.vstack([err1**2., err2**2.]).T
    return X, Xerr

def optimize(X, Xerr, param_name='n_components', param_range=np.array([256, 512, 1024, 2048, 4096, 8182])):
    """
    optimize XD for param_name with the param_range
    """
    shuffle_split = ShuffleSplit(len(X), 16, test_size=0.3)
    train_scores, test_scores = validation_curve(xdgmm, X=X, y=Xerr, param_name='n_components', param_range=param_range, n_jobs=3, cv=shuffle_split, verbose=1)
    np.savez('xdgmm_scores.npz', train_scores=train_scores, test_scores=test_scores)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    dp.plot_val_curve(param_range, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std)
    return train_scores, test_scores

def multiplyGaussians(a, A, b, B):
    """
    multiple the two gaussians N(a, A) and N(b, B) to generate z_c*N(c, C)
    """
    Ainv = np.linalg.inv(A)
    Binv = np.linalg.inv(B)
    C = np.linalg.inv(Ainv + Binv)
    Cinv = np.linalg.inv(C)
    d = len(a)
    c = np.dot(np.dot(C,Ainv),a) + np.dot(np.dot(C,Binv),b)

    exponent = -0.5*(np.dot(np.dot(np.transpose(a),Ainv),a) + \
                     np.dot(np.dot(np.transpose(b),Binv),b) - \
                     np.dot(np.dot(np.transpose(c),Cinv),c))
    z_c= (2*np.pi)**(-d/2.)*np.linalg.det(C)**0.5*np.linalg.det(A)**-0.5*np.linalg.det(B)**-0.5*np.exp(exponent)

    return c, C, z_c

def plotXarrays(minParallaxMAS, maxParallaxMAS, apparentMagnitude, nPosteriorPoints=10000):
    """
    given a min and max parallax in mas, and apparent magnitude calculate the plotting arrays for my funny unit parallax[mas]*10**(0.2*apparent magnitude[mag]) and parallax[mas]
    """
    minabsMagKinda = parallax2absMagKinda(minParallaxMAS, apparentMagnitude)
    maxabsMagKinda = parallax2absMagKinda(maxParallaxMAS, apparentMagnitude)
    xabsMagKinda = np.linspace(minabsMagKinda, maxabsMagKinda, nPosteriorPoints)
    xparallaxMAS = np.linspace(minParallaxMAS, maxParallaxMAS, nPosteriorPoints)
    return xparallaxMAS, xabsMagKinda


def absMagKindaPosterior(xdgmm, ndim, mean, cov, x, projectedDimension=1):
    """
    calculate the posterior of data likelihood mean, cov with prior xdgmm
    """
    allMeans = np.zeros((xdgmm.n_components, ndim))
    allAmps = np.zeros(xdgmm.n_components)
    allCovs = np.zeros((xdgmm.n_components, ndim, ndim))
    summedPosterior = np.zeros(len(x))
    individualPosterior = np.zeros((xdgmm.n_components, nPosteriorPoints))
    for gg in range(xdgmm.n_components):
        #print mean2[dimension], cov2[dimension], xdgmm.mu[gg], xdgmm.V[gg]
        newMean, newCov, newAmp = multiplyGaussians(xdgmm.mu[gg], xdgmm.V[gg], mean, cov)
        newAmp *= xdgmm.weights[gg]
        allMeans[gg] = newMean
        allAmps[gg] = newAmp
        allCovs[gg] = newCov

        summedPosterior += st.gaussian(newMean[projectedDimension], np.sqrt(newCov[projectedDimension, projectedDimension]), x, amplitude=newAmp)
        individualPosterior[gg,:] = st.gaussian(newMean[projectedDimension], np.sqrt(newCov[projectedDimension, projectedDimension]), x, amplitude=newAmp)
    summedPosterior = summedPosterior/np.sum(allAmps)
    return allMeans, allAmps, allCovs, summedPosterior

def posterior2d(means, amps, covs, xbins, ybins, nperGauss=100000., plot=False):
    """
    sample each individual posterior from xd to generate full posterior in 2d
    """
    previousIndex = 0
    magicNumber = -99999.
    nsamples = np.int(np.rint(np.sum(nperGauss*amps/np.max(amps))))
    samplesX = np.zeros(nsamples) + magicNumber
    samplesY = np.zeros(nsamples) + magicNumber

    for gg in range(xdgmm.n_components):
        nextIndex = np.int(np.rint(nperGauss*amps[gg]/np.max(amps)))
        if plot: figAll, axAll = plt.subplots()
        if nextIndex > 0:
            samplesAll = np.random.multivariate_normal(means[gg], covs[gg], nextIndex)
            samplesX[previousIndex:previousIndex+nextIndex] = samplesAll[:,0]
            samplesY[previousIndex:previousIndex+nextIndex] = samplesAll[:,1]

            if plot:
                figOne, axOne = plt.subplots()
                axOne.hist(absMagKinda2Parallax(samplesAll[:,1], apparentMagnitude), bins=100, histtype='step')
                axOne.set_title('The relative amplitude is ' + str(amps[gg]/np.max(amps)))
                axAll.hist(absMagKinda2Parallax(samplesAll[:,1], apparentMagnitude), bins=absMagKinda2Parallax(ybins, apparentMagnitude), histtype='step')
                figOne.savefig('example.' + str(k) + '.eachSample.' + str(gg) + '.png')
        if plot:
            figAll.savefig('example' + str(k) + '.eachSample.png')

        previousIndex = nextIndex

    samplesX = samplesX[samplesX != magicNumber]
    samplesY = samplesY[samplesY != magicNumber]

    #print samplesX, samplesY, np.min(samplesX), np.max(samplesX), np.min(samplesY), np.max(samplesY)

    Z, xedges, yedges = np.histogram2d(samplesX, absMagKinda2absMag(samplesY), bins=[xbins, ybins], normed=True)
    return xedges, yedges, Z, samplesY

def plotPosteriorEachStar(meanPost, covPost, ampPost, summedPosterior, meanLike, covLike, xdgmm, xparallaxMAS, xabsMagKinda, apparentMagnitude, X=None, Y=None, Z=None, plot2D=False, samplesY=None, axAll=None):
    """
    plot the posterior of each star into example.png and feed it axAll to put it on another plot
    """

    ylabel_posterior_logd = r'P(log d | y, $\sigma_{y}$)'
    ylabel_posterior_d = r'P(d | y, $\sigma_{y}$)'
    ylabel_posterior_parallax = r'P($\varpi | y, \sigma_{y}$)'
    ylabel_likelihood_d = r'Likelihood P(y | d, $\sigma_{y}$)'
    ylabel_likelihood_logd = r'Likelihood P(y | log d, $\sigma_{y}$)'

    fig, ax = plt.subplots(1, 4, figsize=(17,7))
    positive = xparallaxMAS > 0.
    for gg in range(xdgmm.n_components):
        if plot2D:
            #plot the contours of the full posterior in 2D
            ax[0].contour(X, Y, Z, cmap=plt.get_cmap('Greys'), linewidths=4)
            ax[0].scatter(X, Y, lw=0, color='black', alpha=0.01, s=1)
            axAll[3].contour(X, Y, Z, cmap=plt.get_cmap('Greys'), linewidths=4)
        else:
            #plot each individual gaussian of the posterior weighted by its amplitdue
            points = drawEllipse.plotvector(meanPost[gg], covPost[gg])
            ax[0].plot(points[0, :], absMagKinda2absMag(points[1,:]), 'k', alpha=ampPost[gg]/np.max(ampPost), lw=0.5)
            if axAll is not None: axAll[3].plot(points[0, :], absMagKinda2absMag(points[1,:]), 'k', alpha=ampPost[gg]/np.max(ampPost), lw=0.5)

        #plot the prior
        points = drawEllipse.plotvector(xdgmm.mu[gg], xdgmm.V[gg])
        ax[0].plot(points[0,:],absMagKinda2absMag(points[1,:]), 'r', lw=0.5, alpha=xdgmm.weights[gg]/np.max(xdgmm.weights))

    #plot the 2D likelihood
    pointsData = drawEllipse.plotvector(meanLike, covLike)
    ax[0].plot(pointsData[0, :], absMagKinda2absMag(pointsData[1,:]), 'g', lw=2)
    if axAll is not None:
        axAll[0].plot(pointsData[0, :], absMagKinda2absMag(pointsData[1,:]), 'g', lw=2)
        axAll[3].plot(pointsData[0, :], absMagKinda2absMag(pointsData[1,:]), 'g', alpha=0.5, lw=2)

    #plot the posterior in parallax space
    parallaxLikelihood = st.gaussian(meanLike[1], np.sqrt(covLike[1,1]), xabsMagKinda)
    parallaxPosterior = summedPosterior*10.**(0.2*apparentMagnitude)
    ampRatio = np.max(parallaxPosterior)/np.max(parallaxLikelihood)
    ax[1].plot(xparallaxMAS, parallaxPosterior, 'k', lw=2)
    ax[1].plot(xparallaxMAS, parallaxLikelihood*ampRatio, 'g', lw=2)
    if axAll is not None:
        axAll[1].plot(xparallaxMAS, parallaxLikelihood*ampRatio, 'g', lw=2, alpha=0.5)
        axAll[4].plot(xparallaxMAS, parallaxPosterior, 'k', lw=2, alpha=0.5)

    #plot historgram of y samples vs true distribution to check my sampling is correct
    if plot2D: ax[1].hist(absMagKinda2Parallax(samplesY, apparentMagnitude), color='black', bins=100, histtype='step', normed=True)

    #plot the posteriorin distance space
    distancePosterior = summedPosterior[positive]*xparallaxMAS[positive]**2.*10.**(0.2*apparentMagnitude)
    parallaxLikelihood_positive = st.gaussian(meanLike[1], np.sqrt(covLike[1,1]), xabsMagKinda)[positive]
    ampRatio = np.max(distancePosterior)/np.max(parallaxLikelihood_positive)
    ax[2].plot(1./xparallaxMAS[positive], distancePosterior, 'k', lw=2)
    ax[2].plot(1./xparallaxMAS[positive], parallaxLikelihood_positive*ampRatio, 'g', lw=2)


    #plot the posterior in log distance space
    logdistancePosterior = summedPosterior[positive]*xparallaxMAS[positive]*10.**(0.2*apparentMagnitude)/np.log10(np.exp(1))
    ampRatio = np.max(logdistancePosterior)/np.max(parallaxLikelihood_positive)
    ax[3].plot(np.log10(1./xparallaxMAS[positive]), logdistancePosterior, 'k', lw=2)
    ax[3].plot(np.log10(1./xparallaxMAS[positive]), parallaxLikelihood_positive*ampRatio, 'g', lw=2, label='likelihood')
    if axAll is not None:
        axAll[2].plot(np.log10(1./xparallaxMAS[positive]), parallaxLikelihood_positive*ampRatio, 'g', lw=2, alpha=0.5)
        axAll[5].plot(np.log10(1./xparallaxMAS[positive]), logdistancePosterior, 'k', lw=2, alpha=0.5)

    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)
    ax[0].set_xlabel(xlabel, fontsize=18)
    ax[0].set_ylabel(ylabel, fontsize=18)

    ax[1].set_xlabel('Parallax [mas]', fontsize=18)
    ax[1].set_xlim(-1, 6)
    ax[1].set_ylabel(ylabel_posterior_parallax)

    #ax[2].set_xscale('log')
    ax[2].set_xlim(0.01, 3)
    ax[2].set_xlabel('Distance [kpc]', fontsize=18)
    ax[2].set_ylabel(ylabel_posterior_d)

    ax[3].set_xlim(np.log10(0.3), np.log10(3))
    ax[3].set_xlabel('log Distance [kpc]', fontsize=18)
    ax[3].set_ylabel(ylabel_posterior_logd)
    plt.legend()
    plt.tight_layout()
    fig.savefig('example.png')


def distanceTest(tgas, nPosteriorPoints, data1, data2, err1, err2, xlim, ylim, plot2DPost=False):
    """
    test posterior accuracy using distances to cluster M67
    """
    indicesM67 = m67indices(tgas, plot=False, db=0.5, dl=0.5)

    #all the plot stuff
    figDist, axDist = plt.subplots(2, 3, figsize=(20, 15))
    axDist = axDist.flatten()
    ylabel_posterior_logd = r'P(log d | y, $\sigma_{y}$)'
    ylabel_posterior_d = r'P(d | y, $\sigma_{y}$)'
    ylabel_posterior_parallax = r'P($\varpi | y, \sigma_{y}$)'
    ylabel_likelihood_d = r'Likelihood P(y | d, $\sigma_{y}$)'
    ylabel_likelihood_logd = r'Likelihood P(y | log d, $\sigma_{y}$)'

    #set up the bins for the 2d posterior
    delta = 0.01
    xbins = np.arange(xlim[0], xlim[1], delta)
    ybins = np.arange(ylim[1], ylim[0], delta)
    x = 0.5*(xbins[1:] + xbins[:-1])
    y = 0.5*(ybins[1:] + ybins[:-1])
    X, Y = np.meshgrid(x, y, indexing='ij')


    #the array for the projected posterior
    summedPosterior = np.zeros((np.sum(indicesM67), nPosteriorPoints))

    #for 2D, the maximum number of samples to take from each gaussian
    nperGauss = 100000.

    #loop through stars in the cluster
    for k, index in enumerate(np.where(indicesM67)[0]): #zip([16], [np.where(indicesM67)[0][16]]): #

        #plotting for each star in the cluster
        windowFactor = 5. #the number of sigma to sample in mas for plotting
        minParallaxMAS = tgas['parallax'][index] - windowFactor*tgas['parallax_error'][index]
        maxParallaxMAS = tgas['parallax'][index] + windowFactor*tgas['parallax_error'][index]
        apparentMagnitude = bandDictionary[absmag]['array'][bandDictionary[absmag]['key']][index]
        xparallaxMAS, xabsMagKinda = plotXarrays(minParallaxMAS, maxParallaxMAS, apparentMagnitude, nPosteriorPoints=nPosteriorPoints)

        #for distances, only want positive parallaxes
        positive = xparallaxMAS > 0.

        meanData, covData = matrixize(data1[index], data2[index], err1[index], err2[index])
        meanData = meanData[0]
        covData = covData[0]
        ndim = 2

        #calculate the posterior, a gaussian for each xdgmm component
        allMeans, allAmps, allCov, summedPosterior[k,:] = absMagKindaPosterior(xdgmm, ndim, meanData, covData, xabsMagKinda)

        #for 2D visual of posterior, sample all the xdgmm component gaussians into 2d historgram
        if plot2DPost: xedges, yedges, Z, samplesY = posterior2d(allMeans, allAmps, allCov, xbins, ybins, nperGauss=nperGauss)
        else:
            X = Y = Z = samplesY = None
        #plot the posterior of each star in example.png and feed it the ax for all the stars to accumualte into one plot
        plotPosteriorEachStar(allMeans, allCov, allAmps, summedPosterior[k,:], meanData, covData, xdgmm, xparallaxMAS, xabsMagKinda, apparentMagnitude, X=X, Y=Y, Z=Z, samplesY=samplesY, axAll=axDist, plot2D=plot2DPost)
        os.rename('example.png', 'example.' + str(k) + '.png')

        #plot prior on plot of all stars but only first loop
        if k == 0:
            for gg in range(xdgmm.n_components):
                points = drawEllipse.plotvector(xdgmm.mu[gg], xdgmm.V[gg])
                axDist[0].plot(points[0,:],absMagKinda2absMag(points[1,:]), 'r', lw=0.5, alpha=xdgmm.weights[gg]/np.max(xdgmm.weights))
                axDist[3].plot(points[0,:],absMagKinda2absMag(points[1,:]), 'r', lw=0.5, alpha=xdgmm.weights[gg]/np.max(xdgmm.weights))

        #check that things seem right
        normalization_parallaxPosterior = scipy.integrate.cumtrapz(summedPosterior[k,:]*10.**(0.2*apparentMagnitude), xparallaxMAS)[-1]
        normalization_distancePosterior = scipy.integrate.cumtrapz(summedPosterior[k,:][positive]*xparallaxMAS[positive]**2.*10.**(0.2*apparentMagnitude), 1./xparallaxMAS[positive])[-1]
        normalization_logdistancePosterior = scipy.integrate.cumtrapz(summedPosterior[k,:][positive]*xparallaxMAS[positive]*10.**(0.2*apparentMagnitude)/np.log10(np.exp(1)), np.log10(1./xparallaxMAS[positive]))[-1]

        #print 'the sum of parallax PDF is: ',normalization_parallaxPosterior
        #print 'the sum of distance PDF is :', normalization_distancePosterior
        #print 'the sum of log distance PDF is :', normalization_logdistancePosterior

    np.save('summedPosteriorM67', summedPosterior)


    axDist[0].set_xlabel(xlabel, fontsize=18)
    axDist[0].set_ylabel(ylabel, fontsize=18)
    axDist[0].set_xlim(xlim)
    axDist[0].set_ylim(ylim)
    axDist[3].set_xlabel(xlabel, fontsize=18)
    axDist[3].set_ylabel(ylabel, fontsize=18)
    axDist[3].set_xlim(xlim)
    axDist[3].set_ylim(ylim)


    axDist[1].axvline(1./0.8, color="b", lw=2)
    axDist[1].axvline(1./0.9, color="b", lw=2)
    axDist[4].axvline(1./0.8, color="b", lw=2)
    axDist[4].axvline(1./0.9, color="b", lw=2)


    axDist[1].set_xlabel('Parallax [mas]')
    axDist[4].set_xlabel('Parallax [mas]')
    axDist[1].set_ylabel(ylabel_likelihood_d, fontsize=18)
    axDist[4].set_ylabel(ylabel_posterior_d, fontsize=18)
    axDist[1].set_xlim(-1, 6)
    axDist[4].set_xlim(-1, 6)

    axDist[2].axvline(np.log10(0.8), color="b", lw=2)
    axDist[2].axvline(np.log10(0.9), color="b", lw=2)
    axDist[5].axvline(np.log10(0.8), color="b", lw=2)
    axDist[5].axvline(np.log10(0.9), color="b", lw=2)


    axDist[2].set_xlabel('log Distance [kpc]')
    axDist[5].set_xlabel('log Distance [kpc]')
    axDist[2].set_ylabel(ylabel_likelihood_d, fontsize=18)
    axDist[5].set_ylabel(ylabel_posterior_logd, fontsize=18)
    axDist[2].set_xlim(np.log10(0.1),np.log10(3))
    axDist[5].set_xlim(np.log10(0.1),np.log10(3))

    plt.tight_layout()
    figDist.savefig('distancesM67.png')

def plotPrior(xdgmm, ax, c='k', lw=1):
    for gg in range(xdgmm.n_components):
        points = drawEllipse.plotvector(xdgmm.mu[gg], xdgmm.V[gg])
        ax[0].plot(points[0,:],testXD.absMagKinda2absMag(points[1,:]), c, lw=lw, alpha=xdgmm.weights[gg]/np.max(xdgmm.weights))


def distanceQuantile(color, absMagKinda, color_err, absMagKinda_err, tgas, xdgmm, distanceFile='distance.npy', quantile=0.05, nDistanceSamples=512, nPosteriorPoints=1000, iter='1st', plotPost=False):
    try:
        data = np.load(distanceFile)
        distance = data['distance']
        print 'distance file is: ', distanceFile
    except IOError:
        print 'distance file does not exist: ', distanceFile
        nstars = len(color)
        sourceID = np.zeros(nstars, dtype='>i8')
        #dustEBV = np.zeros(nstars)
        #dustEBV50 = np.zeros(nstars)
        distance = np.zeros(nstars)
        start = time.time()
        logDistance = np.linspace(-2, 1, nPosteriorPoints)
        xparallaxMAS = 1./10.**logDistance
        positive = xparallaxMAS > 0.

        nMidMin = 0
        nMidPost = 0
        nMidFlatMin = 0
        nMidFlatMax = 0
        nSmallMin = 0
        nSmallMax = 0
        nSmallFlatMin = 0
        nSmallFlatMax = 0
        debug = False
        plotPost = True

        distSmalldata = np.load('distanceQuantiles.128gauss.dQ0.05.6th.2MASS.All.npz.save')
        distSmall = distSmalldata['distance']

        distLargedata = np.load('distanceQuantiles.128gauss.dQ0.5.1st.2MASS.All.npz.save')
        distLarge = distLargedata['distance']

        delta = distLarge - distSmall

        if plotPost: fig, ax = plt.subplots()
        for index in np.where(delta < 0)[0]: #range(nstars):
            if np.mod(index, 10000) == 0.0:
                end = time.time()
                print index, ' took ', str(end - start), 'seconds, projecting will be ', str((end-start)*((nstars-index)/10000.))
                start = time.time()
            #if index == 4491: pdb.set_trace()
            #np.savez('dustCorrection_' + dataFilename, ebv=dustEBV, sourceID=sourceID)
            #calculate parallax-ish posterior for each star
            meanData, covData = matrixize(color[index], absMagKinda[index], color_err[index], absMagKinda_err[index])
            meanData = meanData[0]
            covData = covData[0]
            apparentMagnitude = bandDictionary[absmag]['array'][bandDictionary[absmag]['key']][index]
            xabsMagKinda = parallax2absMagKinda(xparallaxMAS, apparentMagnitude)

            if debug:

                windowFactor = 15. #the number of sigma to sample in mas for plotting
                minParallaxMAS = tgas['parallax'][index] - windowFactor*tgas['parallax_error'][index]
                maxParallaxMAS = tgas['parallax'][index] + windowFactor*tgas['parallax_error'][index]
                xparallaxMAS, xabsMagKinda = plotXarrays(minParallaxMAS, maxParallaxMAS, apparentMagnitude, nPosteriorPoints=nPosteriorPoints)
                xabsMagKinda = xabsMagKinda[::-1]
                xparallaxMAS = xparallaxMAS[::-1]
                positive = xparallaxMAS > 0.
                if np.sum(positive) == 0:
                    print str(index) + ' has no positive distance values'
                    continue
                logDistance = np.log10(1./xparallaxMAS[positive])
            allMeans, allAmps, allCovs, summedPosteriorAbsmagKinda = absMagKindaPosterior(xdgmm, ndim, meanData, covData, xabsMagKinda, projectedDimension=1)
            #normalize prior pdf
            #posteriorDistance = summedPosteriorAbsmagKinda[positive]*xparallaxMAS[positive]**2.*10.**(0.2*apparentMagnitude)
            posteriorLogDistance = summedPosteriorAbsmagKinda[positive]*xparallaxMAS[positive]*10.**(0.2*apparentMagnitude)/np.log10(np.exp(1))

            #distanceIncreasing = distance[::-1]
            #posteriorIncreasingDistance = posteriorDistance[::-1]
            #print minParallaxMAS, maxParallaxMAS, tgas['parallax'][index], tgas['parallax_error'][index]
            #print len(posteriorLogDistance), len(logDistance), np.sum(np.isnan(posteriorLogDistance)), np.sum(np.isnan(logDistance))
            cdf = scipy.integrate.cumtrapz(posteriorLogDistance, x=logDistance)
            cdfInv = scipy.interpolate.interp1d(cdf, 0.5*(logDistance[1:] + logDistance[:-1]))
            minDist = logDistance[0]
            maxDist = logDistance[-1]
            P_minDist = posteriorLogDistance[0]
            P_maxDist = posteriorLogDistance[-1]
            #print minParallaxMAS, maxParallaxMAS, minDist, maxDist, np.max(cdf)

            assert np.sum(np.isnan(posteriorLogDistance)) == 0., 'there are Nans in my posterior ' + str(index)
            absMag = absMagKinda2absMag(absMagKinda[index])
            if np.isnan(absMag): absMag = absMagKinda[index]

            #if the posterior lies well within the distance window then do the right thing
            if np.max(cdf) > 0.95:
                distance[index] = 10.**cdfInv(quantile)
                #if np.mod(index, 10000) == 0.0:
                label = 'posterior is good, log distance is ' + '{0:.2f}'.format(float(cdfInv(quantile)))
                if plotPost:
                    plt.cla()
                    ax.plot(logDistance[:-1], cdf, label=label, lw=2)
                    ax.set_xlabel('log distance [kpc]')
                    ax.set_ylabel('cdf')
                    ax.legend()
                    ax.set_title('$J-K$ ' +  '{0:.2f}'.format(float(color[index])) + ' $M_J$ ' + '{0:.2f}'.format(float(absMag)))
                    fig.savefig('cdfplots/cdf.good.' + str(index) + '.' + iter + '.dQ.' + str(quantile) + '.png')

            #if the posterior lies way outside the distance window [0.01-10] kpc then set distance to which ever side of window has higher probability
            elif np.max(cdf) < quantile:
                #print 'The CDF did not reach above ', str(quantile), ' for ', str(index)
                if P_minDist > P_maxDist: #pdf lies at small distances
                    distance[index] = 10.**minDist
                    label = 'posterior small in range, set to ' + '{0:.2f}'.format(float(distance[index]))
                    print label
                    nSmallMin += 1
                if P_minDist < P_maxDist: #pdf lies at large distances
                    distance[index] = 10.**maxDist
                    label = 'posterior small in range, set to ' + '{0:.2f}'.format(float(distance[index]))
                    nSmallMax += 1
                if P_minDist == P_maxDist: #pdf is flat
                    if tgas['parallax'][index] <= 0:
                        distance[index] = 10.**maxDist
                        nSmallFlatMax += 1
                    else:
                        distance[index] = 10.**minDist
                        nSmallFlatMin += 1
                    label = 'The posterior is flat with value '+ str(P_minDist)+', Dist = ' + '{0:.2f}'.format(float(distance[index]))

                    print label
                if plotPost:
                    plt.cla()
                    ax.plot(logDistance[:-1], cdf, label=label, lw=2)
                    ax.set_xlabel('log distance [kpc]')
                    ax.set_ylabel('cdf')
                    ax.legend()
                    ax.set_title('$J-K$ ' +  '{0:.2f}'.format(float(color[index])) + ' $M_J$ ' + '{0:.2f}'.format(float(absMag)))
                    fig.savefig('cdfplots/cdf.Small.' + str(index) + '.' + iter + '.dQ.' + str(quantile) + '.png')

            #if the posterior lies just on the edge, then if the pdf appears to be rising with distance set to quantile, else set to min distance
            else:
                #print 'The max of the CDF is between ', str(quantile), ' and 0.95 for ', str(index)
                if P_minDist > P_maxDist: #pdf not rising with distance
                    distance[index] = 10.**minDist
                    label = 'posterior is mid in range, set to ' + '{0:.2f}'.format(float(distance[index]))
                    print label
                    nMidMin += 1
                if P_minDist < P_maxDist: #pdf rising with distance
                    distance[index] = 10.**cdfInv(quantile)
                    label = 'posterior is mid in range, set to ' + '{0:.2f}'.format(float(distance[index]))
                    nMidPost += 1
                if P_minDist == P_maxDist: #pdf is flat
                    label= 'The posterior is flat with value ' + str(P_minDist)+', Dist= ' + '{0:.2f}'.format(float(distance[index]))
                    print label
                    if tgas['parallax'][index] <= 0:
                        distance[index] = 10.**maxDist
                        nMidFlatMax += 1
                    else:
                        distance[index] = 10.**minDist
                        nMidFlatMin += 1
                    print label
                    distance[index] = 10.**minDist

                if plotPost:
                    plt.cla()
                    ax.plot(logDistance[:-1], cdf, label=label, lw=2)
                    ax.set_xlabel('log distance [kpc]')
                    ax.set_ylabel('cdf')
                    ax.legend()
                    ax.set_title('$J-K$ ' +  '{0:.2f}'.format(float(color[index])) + ' $M_J$ ' + '{0:.2f}'.format(float(absMag)))
                    fig.savefig('cdfplots/cdf.Mid.' + str(index) + '.' + iter + '.dQ.' + str(quantile) + '.png')
            """
            try:
                distance[index] = cdfInv(quantile)
                distanceMedian[index] = cdfInv(0.5)
            except ValueError:
                print np.max(cdf)
                plt.cla()
                ax.plot(distance[::-1][1:], cdf)
                ax.set_xlabel('distance')
                ax.set_ylabel('cdf')
                fig.savefig('cdfplots/cdf.' + str(index) + '.png')
                #distanceMedian[index] = np.nan
            """
        print 'N mid posterior set to minDist: ', nMidMin
        print 'N mid posterior set to quantil Dist: ', nMidPost
        print 'N mid posterior that are flat, set to minDist: ', nMidFlatMin
        print 'N mid posterior that are flat, set to maxDist: ', nMidFlatMax
        print 'N small posterior set to minDist: ', nSmallMin
        print 'N small posterior set to maxDist: ', nSmallMax
        print 'N small posteriors that are flat, set to minDist: ', nSmallFlatMin
        print 'N small posteriors that are flat, set to maxDist: ', nSmallFlatMax


        np.savez(distanceFile, distance=distance)
    return distance

def dustCorrection(tgas, color, color_err, absMagKinda, absMagKinda_err, xdgmm, quantile=0.05, nDistanceSamples=512, max_samples = 2, plot=False, mode='median', dustFile='dustCorrection', distanceFile = 'distanceQuantiles'):

    try:
        data = np.load(dustFile)
        dustEBV = data['ebv']
        dustEBV50 = data['ebv50']
        sourceID = data['sourceID']
        print 'dust file is: ', dustFile
    except IOError:
        print 'dust file does not exist: ', dustFile
        print 'calculating dust corrections, this may take awhile'
        distance = distanceQuantile(color, absMagKinda, color_err, absMagKinda_err, tgas, distanceFile=distanceFile, quantile=quantile)
        sourceID = tgas['source_id']
        l = tgas['l']*units.deg
        b = tgas['b']*units.deg
        start = time.time()
        dustEBV = st.dust(l, b, distance*units.kpc, mode=mode)
        end = time.time()
        #print 'dust sampling ', str(nDistanceSamples), ' took ',str(end-start), ' seconds for index ', str(i)
        print 'calculating dust took ', str(end - start), ' seconds'
        assert np.sum(np.isnan(dustEBV)) == 0., 'some stars still have Nan for dust'
        np.savez(dustFile, ebv=dustEBV, sourceID=sourceID)
    if plot:
        data = np.load(distanceFile)
        distanceQuantile = data['distanceQuantile']
        distanceQuantile50 = data['distanceQuantile50']
        figHist, axHist = plt.subplots(2, figsize=(7, 10))
        figDust, axDust = plt.subplots(2, figsize=(7, 10))


        dustEBV[dustEBV==0.0] = 1e-5
        dustEBV50[dustEBV50==0.0] = 1e-5
        axDust[0].hist2d(color, np.log10(dustEBV), bins=100, norm=LogNorm(), cmap='Greys')
        #axDust[1].hist2d(color, np.log10(dustEBVMedian), bins=100, norm=LogNorm(), cmap='Greys')
        axHist[0].hist(color, bins=100, histtype='step', log=True, label='5% quantile', lw=2)
        axHist[0].hist(color, bins=100, histtype='step', log=True, label='50% quantile', lw=2)
        axHist[1].hist(np.log10(dustEBV), bins=100, histtype='step', log=True, label='5% quantile', lw=2)
        axHist[1].hist(np.log10(dustEBV50), bins=100, histtype='step', log=True, label='50% quantile', lw=2)
        plt.legend()


        #figDust.colorbar()d

        #axDust[0].set_title('Dust for 0.05 quantile distance')
        #axDust[1].set_title('Dust for 0.5 quantile distance')
        axDust[0].set_xlabel('J-K 5% quantile')
        axHist[0].set_xlabel('J - K')
        axDust[1].set_xlabel('J-K Median')
        axDust[0].set_ylabel('log E(B-V)')
        axDust[1].set_ylabel('log E(B-V)')
        axHist[1].set_xlabel('log E(B-V)')

        figHist.savefig('ebvDistribution1D.png')
        figDust.savefig('ebvDistribution2D.png')

    return dustEBV, sourceID

def dustCorrect(mag, EBV, band):
    """
    using Finkbeiner's dust model, correct the magnitude for dust extinction
    """

    dustCoeff = {'B': 3.626,
                 'V': 2.742,
                 'g': 3.303,
                 'r': 2.285,
                 'i': 1.698,
                 'J': 0.709,
                 'H': 0.449,
                 'K': 0.302}

    return mag - dustCoeff[band]*EBV

def cdf(x, y):
    return scipy.integrate.cumtrapz(x, y)

def samples(x, pdf, N, plot=False):
    randomNumbers = np.random.random(N)
    cdf = scipy.integrate.cumtrapz(pdf, x)[:, None]
    difference = np.abs(cdf - randomNumbers)
    indices = np.where(difference == np.min(difference, axis=0))
    distSamples = x[indices[0]]

    if plot:
        fig, ax = plt.subplots()
        ax.plot(x, pdf)
        ax.plot(x[1:], cdf)
        ax.hist(distSamples, bins=100, normed=True, histtype='step')
        fig.savefig('samples.png')
    return distSamples

def posteriorDistanceAllStars(tgas, nPosteriorPoints, color, absMagKinda, color_err, absMagKinda_err, xdgmm, ndim=2, projectedDimension=1, posteriorFile = 'posteriorDistanceTgas'):
    nstars = len(tgas)
    summedPosterior = np.zeros((nstars, nPosteriorPoints))
    distancePosterior = np.zeros((nstars, nPosteriorPoints))
    colorDustCorrected = np.zeros(nstars)
    absMagDustCorrected = np.zeros(nstars)

    nstars = len(tgas)
    for index in range(nstars):
        if np.mod(index, 10000) == 0.0:
            print index
            np.savez(posteriorFile, posterior=summedPosterior, distance=distancePosterior)


        meanData, covData = matrixize(color[index], absMagKinda[index], color_err[index], absMagKinda_err[index])
        meanData = meanData[0]
        covData = covData[0]
        windowFactor = 5. #the number of sigma to sample in mas for plotting
        minParallaxMAS = tgas['parallax'][index] - windowFactor*tgas['parallax_error'][index]
        maxParallaxMAS = tgas['parallax'][index] + windowFactor*tgas['parallax_error'][index]
        apparentMagnitude = bandDictionary[absmag]['array'][bandDictionary[absmag]['key']][index]
        xparallaxMAS, xabsMagKinda = plotXarrays(minParallaxMAS, maxParallaxMAS, apparentMagnitude, nPosteriorPoints=nPosteriorPoints)

        positive = xparallaxMAS > 0.
        allMeans, allAmps, allCovs, summedPosteriorAbsmagKinda = absMagKindaPosterior(xdgmm, ndim, meanData, covData, xabsMagKinda, projectedDimension=projectedDimension)

        posteriorDistance = summedPosteriorAbsmagKinda[positive]*xparallaxMAS[positive]**2.*10.**(0.2*apparentMagnitude)
        distance = 1./xparallaxMAS[positive]

        summedPosterior[index, :] = summedPosteriorAbsmagKinda*xparallaxMAS**2.*10.**(0.2*apparentMagnitude)
        distancePosterior[index, :] = 1./xparallaxMAS
    sourceID = tgas['source_id']
    np.savez(posteriorFile, posterior=summedPosterior, distance=distancePosterior, sourceID=sourceID)
    return summedPosterior, distancePosterior, sourceID

def correctForDust(tgas, color, color_err, absMagKinda, absMagKinda_err, xdgmm, dustFile='dustCorrection', distanceFile = 'distanceQuantiles', xdgmmFilename='xdgmm'):

    dustEBV, sourceID = dustCorrection(tgas, color, color_err, absMagKinda, absMagKinda_err, xdgmm, quantile=0.05, nDistanceSamples=128, max_samples=1, mode='median', plot=True, distanceFile=distanceFile, dustFile=dustFile)
    print 'dust attenuations calculated'
    #make sure the dust array and tgas arrays are ordered the same
    assert np.sum(tgas['source_id'] - sourceID) == 0.0, 'dust and data arrays are sorted differently !!!'

    #apply dust correction to data
    mag1DustCorrected   = dustCorrect(bandDictionary[mag1]['array']  [bandDictionary[mag1]['key']], dustEBV, mag1)
    mag2DustCorrected   = dustCorrect(bandDictionary[mag2]['array']  [bandDictionary[mag2]['key']], dustEBV, mag2)
    apparentMagnitude = bandDictionary[absmag]['array'][bandDictionary[absmag]['key']]
    apparentMagDustCorrected = dustCorrect(apparentMagnitude, dustEBV, absmag)
    absMagKindaDustCorrected = tgas['parallax']*10.**(0.2*apparentMagDustCorrected)
    dustCorrectedArraysGenerated = True
    #B_dustcorrected = dustCorrection(Apass['bmag'], bayesDust, 'B')
    #need to define color_err and absMagKinda_err when including dust correction
    colorDustCorrected = mag1DustCorrected - mag2DustCorrected

    Q_J = 0.709

    dx_color = np.abs((bandDictionary[mag1]['array']  [bandDictionary[mag1]['key']] -
                       bandDictionary[mag2]['array']  [bandDictionary[mag2]['key']]) -
                      (mag1DustCorrected - mag2DustCorrected))
    dx_shmag = 0.2*np.log(10)*absMagKindaDustCorrected*Q_J*dustEBV
    dx = np.array((dx_color, dx_shmag))
    C_dust = np.dot(dx, dx.T)
    pdb.set_trace()
    #regenerate prior
    X, Xerr = matrixize(colorDustCorrected, absMagKindaDustCorrected, color_err, absMagKinda_err)
    xdgmm.fit(X, Xerr+C_dust)
    xdgmm.save_model(xdgmmFilename)

    sample = xdgmm.sample(Nsamples)
    dp.plot_sample(colorDustCorrected, absMagKinda2absMag(absMagKindaDustCorrected), colorDustCorrected, absMagKinda2absMag(absMagKindaDustCorrected),
                sample[:,0],absMagKinda2absMag(sample[:,1]),xdgmm, xerr=color_err, yerr=absMagKinda2absMag(absMagKinda_err), xlabel=xlabel, ylabel=ylabel)

    return colorDustCorrected, absMagKindaDustCorrected, xdgmm

def colorArray(mag1, mag2, dustEBV, bandDictionary):
    mag1DustCorrected   = dustCorrect(bandDictionary[mag1]['array']  [bandDictionary[mag1]['key']], dustEBV, mag1)
    mag2DustCorrected   = dustCorrect(bandDictionary[mag2]['array']  [bandDictionary[mag2]['key']], dustEBV, mag2)
    return mag1DustCorrected - mag2DustCorrected


def absMagKindaArray(absmag, dustEBV, bandDictionary):
    apparentMagnitude = bandDictionary[absmag]['array'][bandDictionary[absmag]['key']]
    apparentMagDustCorrected = dustCorrect(apparentMagnitude, dustEBV, absmag)
    absMagKindaDustCorrected = tgas['parallax']*10.**(0.2*apparentMagDustCorrected)
    return absMagKindaDustCorrected


def dataArrays(survey='2MASS'):

    tgas = fits.getdata("stacked_tgas.fits", 1)
    Apass = fits.getdata('tgas-matched-apass-dr9.fits')
    twoMass = fits.getdata('tgas-matched-2mass.fits')

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
        ylim = [6, -4]

    bandDictionary = {'B':{'key':'bmag', 'err_key':'e_bmag', 'array':twoMass},
                     'V':{'key':'vmag', 'err_key':'e_vmag', 'array':Apass},
                     'J':{'key':'j_mag', 'err_key':'j_cmsig', 'array':twoMass},
                     'K':{'key':'k_mag', 'err_key':'k_cmsig', 'array':twoMass},
                     'G':{'key':'phot_g_mean_mag', 'array':tgas}}

    nonzeroError = (bandDictionary[mag1]['array'][bandDictionary[mag1]['err_key']] != 0.0) & \
                   (bandDictionary[mag2]['array'][bandDictionary[mag2]['err_key']] != 0.0)

    bayes = BayestarQuery(max_samples=1)
    dust = bayes(SkyCoord(tgas['l']*units.deg, tgas['b']*units.deg, frame='galactic'), mode='median')
    nanDust = np.isnan(dust[:,0])

    nanPhotErr = ~np.isnan(bandDictionary[mag1]['array'][bandDictionary[mag1]['err_key']]) & \
                 ~np.isnan(bandDictionary[mag2]['array'][bandDictionary[mag2]['err_key']]) & \
                 ~np.isnan(bandDictionary[absmag]['array'][bandDictionary[absmag]['err_key']])

    nanPhot = ~np.isnan(bandDictionary[mag1]['array'][bandDictionary[mag1]['key']]) & \
              ~np.isnan(bandDictionary[mag2]['array'][bandDictionary[mag2]['key']]) & \
              ~np.isnan(bandDictionary[absmag]['array'][bandDictionary[absmag]['key']])

    if survey == '2MASS':
        nonZeroColor = (bandDictionary[mag1]['array'][bandDictionary[mag1]['key']] -
                        bandDictionary[mag2]['array'][bandDictionary[mag2]['key']] != 0.0) & \
                       (bandDictionary[mag1]['array'][bandDictionary[mag1]['key']] != 0.0)
        indices = twoMass['matched'] & nonzeroError & ~nanDust & nanPhot & nanPhotErr

    else:
        indices = parallaxSNcut & lowPhotErrorcut & nonzeroError & ~nanDust

    tgas = tgas[indices]
    Apass = Apass[indices]
    twoMass = twoMass[indices]
    bandDictionary = {'B':{'key':'bmag', 'err_key':'e_bmag', 'array':Apass},
                      'V':{'key':'vmag', 'err_key':'e_vmag', 'array':Apass},
                      'J':{'key':'j_mag', 'err_key':'j_cmsig', 'array':twoMass},
                      'K':{'key':'k_mag', 'err_key':'k_cmsig', 'array':twoMass},
                      'G':{'key':'phot_g_mean_mag', 'array':tgas}}

    return tgas, twoMass, Apass, bandDictionary, indices


if __name__ == '__main__':

    survey = '2MASS'        #survey to calculate prior with
    np.random.seed(2)
    #thresholdSN = 0.001     #threshold S/N
    ngauss = np.int(sys.argv[1]) #128            #number of gaussians in the XD
    quantile = np.float(sys.argv[2])

    Nsamples = 120000       #number of samples of the XD to plot
    nPosteriorPoints = 1000 #number of elements in the posterior array
    projectedDimension = 1  #which dimension to project the prior onto
    ndim = 2

    subset = False          #subsample the data to generate the XD prior
    dustCorrectedArraysGenerated = False
    dustEBV = None


    tgas, twoMass, Apass, bandDictionary, indices = dataArrays()

    """
    dataFilename = 'All.npz'
    tgas = fits.getdata("stacked_tgas.fits", 1)
    #tgasRave = fits.getdata('tgas-rave.fits', 1)
    Apass = fits.getdata('tgas-matched-apass-dr9.fits')
    #tgasWise = fits.getdata('tgas-matched-wise.fits')
    twoMass = fits.getdata('tgas-matched-2mass.fits')


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
        ylim = [6, -4]

    bandDictionary = {'B':{'key':'bmag', 'err_key':'e_bmag', 'array':twoMass},
                      'V':{'key':'vmag', 'err_key':'e_vmag', 'array':Apass},
                      'J':{'key':'j_mag', 'err_key':'j_cmsig', 'array':twoMass},
                      'K':{'key':'k_mag', 'err_key':'k_cmsig', 'array':twoMass},
                      'G':{'key':'phot_g_mean_mag', 'array':tgas}}


    #parallaxSNcut = tgas['parallax']/tgas['parallax_error'] >= thresholdSN
    #sigMax = 1.086/thresholdSN
    #lowPhotErrorcut = (bandDictionary[mag1]['array'][bandDictionary[mag1]['err_key']] < sigMax) & \
    #                  (bandDictionary[mag2]['array'][bandDictionary[mag2]['err_key']] < sigMax)
    nonzeroError = (bandDictionary[mag1]['array'][bandDictionary[mag1]['err_key']] != 0.0) & \
                   (bandDictionary[mag2]['array'][bandDictionary[mag2]['err_key']] != 0.0)

    bayes = BayestarQuery(max_samples=1)
    dust = bayes(SkyCoord(tgas['l']*units.deg, tgas['b']*units.deg, frame='galactic'), mode='median')
    nanDust = np.isnan(dust[:,0])

    nanPhotErr = ~np.isnan(bandDictionary[mag1]['array'][bandDictionary[mag1]['err_key']]) & \
              ~np.isnan(bandDictionary[mag2]['array'][bandDictionary[mag2]['err_key']]) & \
              ~np.isnan(bandDictionary[absmag]['array'][bandDictionary[absmag]['err_key']])

    nanPhot = ~np.isnan(bandDictionary[mag1]['array'][bandDictionary[mag1]['key']]) & \
              ~np.isnan(bandDictionary[mag2]['array'][bandDictionary[mag2]['key']]) & \
              ~np.isnan(bandDictionary[absmag]['array'][bandDictionary[absmag]['key']])

    if survey == '2MASS':
        nonZeroColor = (bandDictionary[mag1]['array'][bandDictionary[mag1]['key']] -
                        bandDictionary[mag2]['array'][bandDictionary[mag2]['key']] != 0.0) & \
                       (bandDictionary[mag1]['array'][bandDictionary[mag1]['key']] != 0.0)

        indices = twoMass['matched'] & nonzeroError & ~nanDust & nanPhot & nanPhotErr

    else:
        indices = parallaxSNcut & lowPhotErrorcut & nonzeroError & ~nanDust

    print 'N stars matching all criteria: ', str(np.sum(indices))

    tgas = tgas[indices]
    Apass = Apass[indices]
    twoMass = twoMass[indices]
    bandDictionary = {'B':{'key':'bmag', 'err_key':'e_bmag', 'array':Apass},
                      'V':{'key':'vmag', 'err_key':'e_vmag', 'array':Apass},
                      'J':{'key':'j_mag', 'err_key':'j_cmsig', 'array':twoMass},
                      'K':{'key':'k_mag', 'err_key':'k_cmsig', 'array':twoMass},
                      'G':{'key':'phot_g_mean_mag', 'array':tgas}}
    """

    iteration = ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th']
    previteration = ['0th', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th']

    for iter, previter in zip(iteration, previteration):

        xdgmmFilename = 'xdgmm.'             + str(ngauss) + 'gauss.dQ' + str(quantile) + '.' + iter + '.' + survey + '.' + dataFilename + '.fit'
        distanceFile  = 'distanceQuantiles.' + str(ngauss) + 'gauss.dQ' + str(quantile) + '.' + iter + '.' + survey + '.' + dataFilename
        dustFile      = 'dustCorrection.'    + str(ngauss) + 'gauss.dQ' + str(quantile) + '.' + iter + '.' + survey + '.' + dataFilename
        priorFile     = 'prior.'             + str(ngauss) + 'gauss.dQ' + str(quantile) + '.' + iter + '.' + survey + '.' + dataFilename + '.png'
        dataFile      = 'data.'              + str(ngauss) + 'gauss.dQ' + str(quantile) + '.' + iter + '.' + survey + '.' + dataFilename + '.png'

        if previter == '0th':

            dustZeroFile = 'dustCorrection.128gauss.dQ0.05.5th.2MASS.All.npz'
            data = np.load(dustZeroFile)
            dustEBV = data['ebv']
            #dustEBV = np.zeros(np.sum(indices))

        else:
            if not isinstance(dustEBV,np.ndarray):
                dustFilePrev = 'dustCorrection.'    + str(ngauss) + 'gauss.dQ' +str(quantile) + '.' + previter + '.' + survey + '.' + dataFilename
                data = np.load(dustFilePrev)
                dustEBV = data['ebv']
            assert np.sum(dustEBV) != 0.0, 'dust for iteration ' + str(iter) +  ' not read in properly'

        color = colorArray(mag1, mag2, dustEBV, bandDictionary)
        absMagKinda = absMagKindaArray(absmag, dustEBV, bandDictionary)

        color_err = np.sqrt(bandDictionary[mag1]['array'][bandDictionary[mag1]['err_key']]**2. + bandDictionary[mag2]['array'][bandDictionary[mag2]['err_key']]**2.)
        absMagKinda_err = tgas['parallax_error']*10.**(0.2*bandDictionary[absmag]['array'][bandDictionary[absmag]['key']])

        try:

            xdgmm = XDGMM(filename=xdgmmFilename)
            print 'dust corrected XD read in for iteration ', iter

        except IOError:
            print 'generating XD for iteration ', iter , ' filename= ', xdgmmFilename
            if subset:
                X, Xerr = subset(color, absMagKinda, color_err, absMagKinda_err, nsamples=1024)
            else:
                X, Xerr = matrixize(color, absMagKinda, color_err, absMagKinda_err)

            #add dust uncertainties to covariances
            addDustCov = False
            if addDustCov:
                Q_J = 0.709
                Q_K = 0.302
                dx_color = np.abs(Q_K - Q_J)*dustEBV
                dx_shmag = 0.2*np.log(10)*absMagKindaDustCorrected*Q_J*dustEBV
                dx = np.array((dx_color, dx_shmag))
                C_dust = np.dot(dx, dx.T)
                Xerr += C_dust

            xdgmm = XDGMM(method='Bovy')
            xdgmm.n_components = ngauss
            xdgmm = xdgmm.fit(X, Xerr)
            xdgmm.save_model(xdgmmFilename)


        #plot XD prior
        #figPrior, axPrior = plt.subplots()
        #for gg in range(xdgmm.n_components):
        #    points = drawEllipse.plotvector(xdgmm.mu[gg], xdgmm.V[gg])
        #    axPrior.plot(points[0,:],absMagKinda2absMag(points[1,:]), 'r', lw=0.5, alpha=xdgmm.weights[gg]/np.max(xdgmm.weights))
        #    axPrior.invert_yaxis()
        #figPrior.savefig('prior.' + iter + '.png')

        #plot 2x2 visual of prior w/ samples
        sample = xdgmm.sample(Nsamples)
        dp.plot_sample(color, absMagKinda2absMag(absMagKinda), color, absMagKinda2absMag(absMagKinda),
                    sample[:,0],absMagKinda2absMag(sample[:,1]),xdgmm, xerr=color_err, yerr=absMagKinda2absMag(absMagKinda_err), xlabel=xlabel, ylabel=ylabel)
        os.rename('plot_sample.data.png', dataFile)
        os.rename('plot_sample.prior.png', priorFile)

        #using prior calculate distances

        distance = distanceQuantile(color, absMagKinda, color_err, absMagKinda_err, tgas, xdgmm, distanceFile=distanceFile, quantile=quantile, nDistanceSamples=128, nPosteriorPoints=nPosteriorPoints)


        #using distance, calculate dust
        try:
            data = np.load(dustFile)
            dustEBV = data['ebv']
        except IOError:
            sourceID = tgas['source_id']
            l = tgas['l']*units.deg
            b = tgas['b']*units.deg
            dustEBV = st.dust(l, b, distance*units.kpc, mode='median')
            assert np.sum(np.isnan(dustEBV)) == 0., 'some stars still have Nan for dust'
            np.savez(dustFile, ebv=dustEBV, sourceID=sourceID)



    color = colorArray(mag1, mag2, dustEBV, bandDictionary)
    absMagKinda = absMagKindaArray(absmag, dustEBV, bandDictionary)

    color_err = np.sqrt(bandDictionary[mag1]['array'][bandDictionary[mag1]['err_key']]**2. + bandDictionary[mag2]['array'][bandDictionary[mag2]['err_key']]**2.)
    absMagKinda_err = tgas['parallax_error']*10.**(0.2*bandDictionary[absmag]['array'][bandDictionary[absmag]['key']])

    #check it's working by inferring distances to M67
    distanceTest(tgas, nPosteriorPoints, color, absMagKinda, color_err, absMagKinda_err, xlim, ylim, plot2DPost=False)

    #calculate parallax-ish posterior for each star
    summedPosterior, distancePosterior, sourceID = posteriorDistanceAllStars(tgas, nPosteriorPoints, color, absMagKinda, color_err, absMagKinda_err, xdgmm, ndim=ndim, projectedDimension=projectedDimension, posteriorFile = 'posteriorDistanceTgas_' + str(ngauss) + '_' + dataFilename)
