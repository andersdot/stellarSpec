import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import os
import stellarTwins as st
from extreme_deconvolution import extreme_deconvolution as ed
import astropy.units as units
import numpy as np
import pdb
import time
from sklearn.model_selection import train_test_split

from xdgmm import XDGMM
from sklearn.learning_curve import validation_curve
from sklearn.cross_validation import ShuffleSplit
import demo_plots as dp
import drawEllipse
from astropy.coordinates import SkyCoord
import astropy.units as units
import scipy.integrate
import time

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


def distanceTest(tgasCutMatched, nPosteriorPoints, data1, data2, err1, err2, xlim, ylim, plot2DPost=False):
    """
    test posterior accuracy using distances to cluster M67
    """
    indicesM67 = m67indices(tgasCutMatched, plot=False, db=0.5, dl=0.5)

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
        minParallaxMAS = tgasCutMatched['parallax'][index] - windowFactor*tgasCutMatched['parallax_error'][index]
        maxParallaxMAS = tgasCutMatched['parallax'][index] + windowFactor*tgasCutMatched['parallax_error'][index]
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

def dustCorrectionPrior(tgasCutMatched, dataFilename, quantile=0.05, nDistanceSamples=512, max_samples = 2):
    dustFile = 'dustCorrection_' + ngauss + '_' + dataFilename
    distanceFile = 'distanceQuantiles_' + ngauss + '_' +dataFilename
    try:
        data = np.load(dustFile)
        dustEBV = data['ebv']
        sourceID = data['sourceID']
    except IOError:
        print 'calculating dust corrections, this may take awhile'
        try:
            data = np.load(distanceFile)
            distanceQuantile = data['distanceQuantile']
            distanceQuantile50 = data['distanceQuantile50']
        except IOError:
            nstars = len(tgasCutMatched)
            sourceID = np.zeros(nstars, dtype='>i8')
            #dustEBV = np.zeros(nstars)
            #dustEBV50 = np.zeros(nstars)
            distanceQuantile = np.zeros(nstars)
            distanceQuantile50 = np.zeros(nstars)
            start = time.time()
            for i, index in enumerate(np.where(indices)[0]):
                if np.mod(i, 1000) == 0.0:
                    end = time.time()
                    print i, ' took ', str(end - start), 'seconds, projecting will be ', str((end-start)*(nstars/1000.))
                    start = time.time()

                #np.savez('dustCorrection_' + dataFilename, ebv=dustEBV, sourceID=sourceID)

                #calculate parallax-ish posterior for each star
                meanData, covData = matrixize(color[index], absMagKinda[index], color_err[index], absMagKinda_err[index])
                meanData = meanData[0]
                covData = covData[0]
                windowFactor = 5. #the number of sigma to sample in mas for plotting
                minParallaxMAS = tgasCutMatched['parallax'][index] - windowFactor*tgasCutMatched['parallax_error'][index]
                maxParallaxMAS = tgasCutMatched['parallax'][index] + windowFactor*tgasCutMatched['parallax_error'][index]
                apparentMagnitude = bandDictionary[absmag]['array'][bandDictionary[absmag]['key']][index]
                xparallaxMAS, xabsMagKinda = plotXarrays(minParallaxMAS, maxParallaxMAS, apparentMagnitude, nPosteriorPoints=nPosteriorPoints)

                positive = xparallaxMAS > 0.

                allMeans, allAmps, allCovs, summedPosteriorAbsmagKinda = absMagKindaPosterior(xdgmm, ndim, meanData, covData, xabsMagKinda, projectedDimension=1)

                #normalize prior pdf
                posteriorDistance = summedPosteriorAbsmagKinda[positive]*xparallaxMAS[positive]**2.*10.**(0.2*apparentMagnitude)
                distance = 1./xparallaxMAS[positive]

                #sample the PDF nDistanceSamples
                sampleDistance = samples(distance[::-1], posteriorDistance[::-1], nDistanceSamples, plot=False)

                #find the distance at the 5% quantile
                distanceQuantile[i] = np.percentile(sampleDistance, quantile*100.)
                distanceQuantile50[i] = np.percentile(sampleDistance, 0.5*100.)
            np.savez(distanceFile, distanceQuantile=distanceQuantile, distanceQuantile50=distanceQuantile50)
        sourceID = tgasCutMatched['source_id'][indices]
        l = tgasCutMatched['l'][indices]*units.deg
        b = tgasCutMatched['b'][indices]*units.deg
        start = time.time()
        dustEBV, dustEBV50 = st.dust([l,l], [b,b], [distanceQuantile*units.kpc, distanceQuantile50*units.kpc], mode='median')
        nan = np.isnan(dustEBV)
        dustNan = st.dust(l[nan], b[nan], None, model='sfd')
        dustEBV[nan] = dustNan
        dustEBV50[nan] = dustNan
        end = time.time()
        #print 'dust sampling ', str(nDistanceSamples), ' took ',str(end-start), ' seconds for index ', str(i)

        np.savez(dustFile, ebv=dustEBV, sourceID=sourceID, ebv50=dustEBV50)
    return dustEBV, sourceID

def dustCorrection(mag, EBV, band):
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

def posteriorDistanceAllStars(tgasCutMatched, nPosteriorPoints, color, absMagKinda, color_err, absMagKinda_err, xdgmm, ndim=2, projectedDimension=1):
    nstars = len(tgasCutMatched)
    summedPosterior = np.zeros((nstars, nPosteriorPoints))
    distancePosterior = np.zeros((nstars, nPosteriorPoints))
    colorDustCorrected = np.zeros(nstars)
    absMagDustCorrected = np.zeros(nstars)
    posteriorFile = 'posteriorDistanceTgas_' + ngauss + '_' + dataFilename

    for i, index in enumerate(np.where(indices)[0]):
        if np.mod(i, 1000) == 0.0:
            print i
            np.savez(posteriorFile, posterior=summedPosterior, distance=distancePosterior, sourceID=sourceID)


        meanData, covData = matrixize(color[index], absMagKinda[index], color_err[index], absMagKinda_err[index])
        meanData = meanData[0]
        covData = covData[0]
        windowFactor = 5. #the number of sigma to sample in mas for plotting
        minParallaxMAS = tgasCutMatched['parallax'][index] - windowFactor*tgasCutMatched['parallax_error'][index]
        maxParallaxMAS = tgasCutMatched['parallax'][index] + windowFactor*tgasCutMatched['parallax_error'][index]
        apparentMagnitude = bandDictionary[absmag]['array'][bandDictionary[absmag]['key']][index]
        xparallaxMAS, xabsMagKinda = plotXarrays(minParallaxMAS, maxParallaxMAS, apparentMagnitude, nPosteriorPoints=nPosteriorPoints)

        positive = xparallaxMAS > 0.
        allMeans, allAmps, allCovs, summedPosteriorAbsmagKinda = absMagKindaPosterior(xdgmm, ndim, meanData, covData, xabsMagKinda, projectedDimension=projectedDimension)

        posteriorDistance = summedPosteriorAbsmagKinda[positive]*xparallaxMAS[positive]**2.*10.**(0.2*apparentMagnitude)
        distance = 1./xparallaxMAS[positive]

        summedPosterior[i, :] = summedPosteriorAbsmagKinda*xparallaxMAS**2.*10.**(0.2*apparentMagnitude)
        distancePosterior[i, :] = 1./xparallaxMAS
        sourceID[i] = tgasCutMatched['source_id'][index]

    np.savez(posteriorFile, posterior=summedPosterior, distance=distancePosterior, sourceID=sourceID)
    return summedPosterior, distancePosterior, sourceID

if __name__ == '__main__':

    survey = '2MASS'
    np.random.seed(2)
    thresholdSN = 0.001
    ngauss = 16
    nstar = '1.2M'
    Nsamples = 120000
    nPosteriorPoints = 1000
    projectedDimension = 1
    ndim = 2
    #dataFilename = 'cutMatchedArrays.SN0.001.npz'
    #dataFilename = 'cutMatchedArrays.tgasApassSN0.npz'
    #dataFilename = 'cutMatchedArrays.1_20_100_4.npz'
    dataFilename = 'cutMatchedArrays.SN0.001.npz'
    xdgmmFilename = 'xdgmm.'+ str(ngauss) + 'gauss.'+dataFilename+'.' + survey + '.fit'
    xdgmmFilenameDust = 'xdgmm.'+ str(ngauss) + 'gauss.'+dataFilename+'.' + survey + '.dustCorrected.fit'
    useDust = False
    optimize = False
    subset = False
    timing = False
    dustCorrectedArraysGenerated = False

    try:
        cutMatchedArrays  = np.load(dataFilename)
        tgasCutMatched    = cutMatchedArrays['tgasCutMatched']
        apassCutMatched   = cutMatchedArrays['apassCutMatched']
        #raveCutMatched    = cutMatchedArrays['raveCutMatched']
        twoMassCutMatched = cutMatchedArrays['twoMassCutMatched']
        #wiseCutMatched    = cutMatchedArrays['wiseCutMatched']
        #distCutMatched    = cutMatchedArrays['distCutMatched']
    except IOError:
        tgasCutMatched, apassCutMatched, raveCutMatched, twoMassCutMatched, wiseCutMatched, distCutMatched = st.observationsCutMatched(SNthreshold=thresholdSN, filename=dataFilename)
    print 'Number of Matched stars is: ', len(tgasCutMatched)

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

    bandDictionary = {'B':{'key':'bmag', 'err_key':'e_bmag', 'array':apassCutMatched},
                      'V':{'key':'vmag', 'err_key':'e_vmag', 'array':apassCutMatched},
                      'J':{'key':'j_mag', 'err_key':'j_cmsig', 'array':twoMassCutMatched},
                      'K':{'key':'k_mag', 'err_key':'k_cmsig', 'array':twoMassCutMatched},
                      'G':{'key':'phot_g_mean_mag', 'array':tgasCutMatched}}


    if useDust:
        bayesDust = st.dust(tgasCutMatched['l']*units.deg, tgasCutMatched['b']*units.deg, np.median(distCutMatched, axis=1)*units.pc)
        mag1DustCorrected   = dustCorrection(bandDictionary[mag1]['array']  [bandDictionary[mag1]['key']], bayesDust, mag1)
        mag2DustCorrected   = dustCorrection(bandDictionary[mag2]['array']  [bandDictionary[mag2]['key']], bayesDust, mag2)
        absMagDustCorrected = dustCorrection(bandDictionary[absmag]['array'][bandDictionary[absmag]['key']], bayesDust, absmag)
        #B_dustcorrected = dustCorrection(apassCutMatched['bmag'], bayesDust, 'B')
        #need to define color_err and absMagKinda_err when including dust correction
        color = mag1DustCorrected - mag2DustCorrected
    else:
        color = bandDictionary[mag1]['array'][bandDictionary[mag1]['key']] - \
                bandDictionary[mag2]['array'][bandDictionary[mag2]['key']]
        color_err = np.sqrt(bandDictionary[mag1]['array'][bandDictionary[mag1]['err_key']]**2. + bandDictionary[mag2]['array'][bandDictionary[mag2]['err_key']]**2.)
        absMagKinda = tgasCutMatched['parallax']*10.**(0.2*bandDictionary[absmag]['array'][bandDictionary[absmag]['key']])
        absMagKinda_err = tgasCutMatched['parallax_error']*10.**(0.2*bandDictionary[absmag]['array'][bandDictionary[absmag]['key']])

    data1 = color
    data2 = absMagKinda
    err1 = color_err
    err2 = absMagKinda_err

    parallaxSNcut = tgasCutMatched['parallax']/tgasCutMatched['parallax_error'] >= thresholdSN
    sigMax = 1.086/thresholdSN
    lowPhotErrorcut = (bandDictionary[mag1]['array'][bandDictionary[mag1]['err_key']] < sigMax) & \
                      (bandDictionary[mag2]['array'][bandDictionary[mag2]['err_key']] < sigMax)
    nonzeroError = (bandDictionary[mag1]['array'][bandDictionary[mag1]['err_key']] != 0.0) & \
                   (bandDictionary[mag2]['array'][bandDictionary[mag2]['err_key']] != 0.0)

    if survey == '2MASS':
        nonZeroColor = (bandDictionary[mag1]['array'][bandDictionary[mag1]['key']] -
                        bandDictionary[mag2]['array'][bandDictionary[mag2]['key']] != 0.0) & \
                       (bandDictionary[mag1]['array'][bandDictionary[mag1]['key']] != 0.0)

        indices = parallaxSNcut & lowPhotErrorcut & nonZeroColor & nonzeroError

    else:
        indices = parallaxSNcut & lowPhotErrorcut & nonzeroError

    try:
        xdgmm = XDGMM(filename=xdgmmFilenameDust)
        print 'dust corrected XD read in'
    except IOError:
        try:
            xdgmm = XDGMM(filename=xdgmmFilename)
            print 'non corrected XD read in'
        except IOError:
            if subset:
                X, Xerr = subset(data1[indices], data2[indices], err1[indices], err2[indices], nsamples=1024)
            else:
                X, Xerr = matrixize(data1[indices], data2[indices], err1[indices], err2[indices])
            xdgmm = XDGMM(method='Bovy')
            xdgmm.n_components = ngauss
            xdgmm = xdgmm.fit(X, Xerr)
            xdgmm.save_model(xdgmmFilename)


        #plot XD prior
        figPrior, axPrior = plt.subplots()
        for gg in range(xdgmm.n_components):
            points = drawEllipse.plotvector(xdgmm.mu[gg], xdgmm.V[gg])
            axPrior.plot(points[0,:],absMagKinda2absMag(points[1,:]), 'r', lw=0.5, alpha=xdgmm.weights[gg]/np.max(xdgmm.weights))
            axPrior.invert_yaxis()
        figPrior.savefig('prior.png')

        #plot 2x2 visual of prior w/ samples
        sample = xdgmm.sample(Nsamples)
        dp.plot_sample(data1[indices], absMagKinda2absMag(data2[indices]), data1[indices], absMagKinda2absMag(data2[indices]),
                    sample[:,0],absMagKinda2absMag(sample[:,1]),xdgmm, xerr=err1[indices], yerr=absMagKinda2absMag(err2[indices]), xlabel=xlabel, ylabel=ylabel)
        os.rename('plot_sample.png', 'prior.ngauss'+str(ngauss)+'.' + dataFilename + '.' + survey + '.png')

        dustEBV, sourceID = dustCorrectionPrior(tgasCutMatched, dataFilename, quantile=0.05, nDistanceSamples=128, max_samples=None)

        assert np.sum(tgasCutMatched['source_id'][indices] - sourceID) == 0.0, 'dust and data arrays are sorted differently !!!'

        mag1DustCorrected   = dustCorrection(bandDictionary[mag1]['array']  [bandDictionary[mag1]['key']][indices], dustEBV, mag1)
        mag2DustCorrected   = dustCorrection(bandDictionary[mag2]['array']  [bandDictionary[mag2]['key']][indices], dustEBV, mag2)
        apparentMagnitude = bandDictionary[absmag]['array'][bandDictionary[absmag]['key']][indices]
        apparentMagDustCorrected = dustCorrection(apparentMagnitude, dustEBV, absmag)
        absMagKindaDustCorrected = tgasCutMatched['parallax'][indices]*10.**(0.2*apparentMagDustCorrected)
        dustCorrectedArraysGenerated = True
        #B_dustcorrected = dustCorrection(apassCutMatched['bmag'], bayesDust, 'B')
        #need to define color_err and absMagKinda_err when including dust correction
        colorDustCorrected = mag1DustCorrected - mag2DustCorrected
        data1 = colorDustCorrected
        data2 = absMagKindaDustCorrected
        X, Xerr = matrixize(data1, data2, err1, err2)
        xdgmm.fit(X, Xerr)
        xdgmm.save_model(xdgmmFilenameDust)
    if not dustCorrectedArraysGenerated:
        dustEBV, sourceID = dustCorrectionPrior(tgasCutMatched, dataFilename, quantile=0.05, nDistanceSamples=128, max_samples=None)
        mag1DustCorrected   = dustCorrection(bandDictionary[mag1]['array']  [bandDictionary[mag1]['key']][indices], dustEBV, mag1)
        mag2DustCorrected   = dustCorrection(bandDictionary[mag2]['array']  [bandDictionary[mag2]['key']][indices], dustEBV, mag2)
        apparentMagnitude = bandDictionary[absmag]['array'][bandDictionary[absmag]['key']][indices]
        apparentMagDustCorrected = dustCorrection(apparentMagnitude, dustEBV, absmag)
        absMagKindaDustCorrected = tgasCutMatched['parallax'][indices]*10.**(0.2*apparentMagDustCorrected)
        #B_dustcorrected = dustCorrection(apassCutMatched['bmag'], bayesDust, 'B')
        #need to define color_err and absMagKinda_err when including dust correction
        colorDustCorrected = mag1DustCorrected - mag2DustCorrected

        pdb.set_trace()
    sample = xdgmm.sample(Nsamples)
    dp.plot_sample(colorDustCorrected, absMagKinda2absMag(absMagKindaDustCorrected), colorDustCorrected, absMagKinda2absMag(absMagKindaDustCorrected),
                sample[:,0],absMagKinda2absMag(sample[:,1]),xdgmm, xerr=err1[indices], yerr=absMagKinda2absMag(err2[indices]), xlabel=xlabel, ylabel=ylabel)
    os.rename('plot_sample.png', 'prior.ngauss'+str(ngauss)+'.' + dataFilename + '.' + survey + '.dustCorrected.png')


    #check it's working by inferring distances to M67
    #distanceTest(tgasCutMatched, nPosteriorPoints, data1, data2, err1, err2, xlim, ylim, plot2DPost=False)

    #calculate parallax-ish posterior for each star

    summedPosterior, distancePosterior, sourceID = posteriorDistanceAllStars(tgasCutMatched, nPosteriorPoints, colorDustCorrected, absMagKindaDustCorrected, color_err, absMagKinda_err, xdgmm, ndim=ndim, projectedDimension=projectedDimension)
