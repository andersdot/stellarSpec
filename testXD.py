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
    #print 'C is: ', C, 'A is: ', A, 'Ainv is: ', Ainv, 'B is: ', B, 'Binv is: ', Binv
    #print 'the first term is : mean - ', a, ' first part - ', np.dot(C, Ainv), ' whole thing -', np.dot(np.dot(C,Ainv), a)
    #print 'the second term is: mean - ', b, ' first part - ', np.dot(C, Binv), ' whole thing -', np.dot(np.dot(C,Binv), b)
    c = np.dot(np.dot(C,Ainv),a) + np.dot(np.dot(C,Binv),b)
    #print 'the new mean is :', c

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


def dustCorrection(magnitude, EBV, band):
    """
    using Finkbeiner's dust model, correct the magnitude for dust extinction
    """

    dustCoeff = {'B': 3.626,
                 'V': 2.742,
                 'g': 3.303,
                 'r': 2.285,
                 'i': 1.698}
    return mag - dustCoeff[band]*EBV

if __name__ == '__main__':

    np.random.seed(2)
    thresholdSN = 0.001
    ngauss = 1024
    nstar = '1.2M'
    Nsamples = 120000
    nPosteriorPoints = 10000
    projectedDimension = 1

    dataFilename = 'cutMatchedArrays.tgasApassSN0.npz'
    xdgmmFilename = 'xdgmm.'+ str(ngauss) + 'gauss.'+nstar+ '.SN' + str(thresholdSN) + '.2MASS.fit'

    useDust = False
    optimize = False
    subset = False
    timing = False


    try:
        cutMatchedArrays  = np.load(dataFilename)
        tgasCutMatched    = cutMatchedArrays['tgasCutMatched']
        apassCutMatched   = cutMatchedArrays['apassCutMatched']
        raveCutMatched    = cutMatchedArrays['raveCutMatched']
        twoMassCutMatched = cutMatchedArrays['twoMassCutMatched']
        wiseCutMatched    = cutMatchedArrays['wiseCutMatched']
        distCutMatched    = cutMatchedArrays['distCutMatched']
    except IOError:
        maxlogg = 20
        minlogg = 1
        mintemp = 100
        tgasCutMatched, apassCutMatched, raveCutMatched, twoMassCutMatched, wiseCutMatched, distCutMatched = st.observationsCutMatched(maxlogg=maxlogg, minlogg=minlogg, mintemp=mintemp, SNthreshold=thresholdSN, filename=dataFilename)
    print 'Number of Matched stars is: ', len(tgasCutMatched)


    indicesM67 = m67indices(tgasCutMatched, plot=False, db=0.5, dl=0.5)


    if survey == 'APASS':
        mag1 = 'B'
        mag2 = 'V'
        absmag = 'G'
        xlabel='B-V'
        ylabel = r'M$_\mathrm{G}$'

    if survey == '2MASS':
        mag1 = 'J'
        mag2 = 'K'
        absmag = 'J'
        xlabel = 'J-K$_s$'
        ylabel = r'M$_\mathrm{J}$'

    bandDictionary = {'B':{key:'bmag', errKey:'e_bmag', array:'apassCutMatched'},
                      'V':{key:'vmag', errKey:'e_vmag', array:'apassCutMatched'},
                      'J':{key:'j_mag', errKey:'j_cmsig', array:'twoMassCutMatched'}
                      'K':{key:'k_mag', errKey:'k_cmsig', array:'twoMassCutMatched'}
                      'G':{key:'phot_g_mean_mag', array:'tgasCutMatched'}}


    if useDust:
        bayesDust = st.dust(tgasCutMatched['l']*units.deg, tgasCutMatched['b']*units.deg, np.median(distCutMatched, axis=1)*units.pc)
        mag1DustCorrected   = dustCorrection(bandDictionary[mag1][array]  [bandDictionary[mag1][key]], bayesDust, mag1) 
        mag2DustCorrected   = dustCorrection(bandDictionary[mag2][array]  [bandDictionary[mag2][key]], bayesDust, mag2)
        absMagDustCorrected = dustCorrection(bandDictionary[absmag][array][bandDictionary[absmag][key]], bayesDust, absmag)
        #B_dustcorrected = dustCorrection(apassCutMatched['bmag'], bayesDust, 'B')
        color = mag1DustCorrected - mag2DustCorrected
    else:
        color = bandDictionary[mag1][array][bandDictionary[mag1][key]] - 
                bandDictionary[mag2][array][bandDictionary[mag2][key]]
    color_err = np.sqrt(bandDictionary[mag1][array][bandDictionary[mag1][err_key]]**2. - bandDictionary[mag2][array][bandDictionary[mag2][err_key]]**2.)

    absMagKinda = tgasCutMatched['parallax']*10.**(0.2*bandDictionary[absmag][array][bandDictionary[absmag][key]])
    absMagKinda_err = tgasCutMatched['parallax']*10.**(0.2*bandDictionary[absmag][array][bandDictionary[absmag][errkey]])
                                                                                     
    data1 = color
    data2 = absMagKinda
    err1 = color_err
    err2 = absMagKinda_err

    ylabel_posterior_logd = r'P(log d|$\varpi, \sigma_{\varpi}$)'
    ylabel_posterior_d = r'P(d|$\varpi, \sigma_{\varpi}$)'
    ylabel_posterior_parallax = r'P($\varpi_\mathrm{true}|\varpi, \sigma_{\varpi}$)'
    ylabel_likelihood_d = r'P($\varpi$|d, $\sigma_{\varpi}$)'
    ylabel_likelihood_logd = r'P($\varpi$|log d, $\sigma_{\varpi}$)'

    summedPosterior = np.zeros((np.sum(indicesM67), nPosteriorPoints))

    parallaxSNcut = tgasCutMatched['parallax']/tgasCutMatched['parallax_error'] >= thresholdSN
    sigMax = 1.086/thresholdSN
    lowPhotErrorcut = (apassCutMatched[bandDictionary[mag1][array][bandDictionary[mag1][err_key]]] < sigMax) & (apassCutMatched[bandDictionary[mag2][array][bandDictionary[mag2][err_key]]] < sigMax)

    indices = parallaxSNcut & lowPhotErrorcut

    
    figDist, axDist = plt.subplots(2, 2, figsize=(15, 15))

    axDist = axDist.flatten()

    figDistLin, axDistLin = plt.subplots(2, figsize=(7.5, 10), sharex=True)

    try:
        xdgmm = XDGMM(filename=xdgmmFilename)
    except IOError:
        if subset:
            X, Xerr = subset(data1[indices], data2[indices], err1[indices], err2[indices], nsamples=1024)
        else:
            X, Xerr = matrixize(data1[indices], data2[indices], err1[indices], err2[indices])

        xdgmm = XDGMM(method='Bovy')
        xdgmm.n_components = ngauss
        xdgmm = xdgmm.fit(X, Xerr)
        xdgmm.save_model(xdgmmFilename)
        sample = xdgmm.sample(Nsamples)
        figPrior, axPrior = plt.subplots()
        for gg in range(xdgmm.n_components):
            points = drawEllipse.plotvector(xdgmm.mu[gg], xdgmm.V[gg])
            axPrior.plot(points[0,:],absMagKinda2absMag(points[1,:]), 'r', lw=0.5, alpha=xdgmm.weights[gg]/np.max(xdgmm.weights))
            axPrior.invert_yaxis()
        figPrior.savefig('prior.png')


        dp.plot_sample(data1[indices], absMagKinda2absMag(data2[indices]), data1[indices], absMagKinda2absMag(data2[indices]),
                       sample[:,0],absMagKinda2absMag(sample[:,1]),xdgmm, xerr=err1[indices], yerr=absMagKinda2absMag(err2[indices]), xlabel=xlabel, ylabel=ylabel)
        
        os.rename('plot_sample.png', 'plot_sample_ngauss'+str(ngauss)+'.SN'+str(thresholdSN) + '.2Mass.png')



    for k, index in enumerate(np.where(indicesM67)[0]):
        individualPosterior = np.zeros((xdgmm.n_components, nPosteriorPoints))
        figPost, axPost = plt.subplots(1, 4, figsize=(17,7))

        windowFactor = 5.
        minParallaxMAS = tgasCutMatched['parallax'][index] - windowFactor*tgasCutMatched['parallax_error'][index]
        maxParallaxMAS = tgasCutMatched['parallax'][index] + windowFactor*tgasCutMatched['parallax_error'][index]
        apparentMagnitude = tgasCutMatched['phot_g_mean_mag'][index]
        xparallaxMAS, xabsMagKinda = plotXarrays(minParallaxMAS, maxParallaxMAS, apparentMagnitude, nPosteriorPoints=nPosteriorPoints)

        positive = xparallaxMAS > 0.
        print 'the min and max of xparallax is: ', np.min(xparallaxMAS), np.max(xparallaxMAS)
        print 'the measured parallax is: ', tgasCutMatched['parallax'][index]

        dimension = 0
        mean2, cov2 = matrixize(data1[index], data2[index], err1[index], err2[index])

        ndim = 2
        allMeans = np.zeros((xdgmm.n_components, ndim))
        allAmps = np.zeros(xdgmm.n_components)
        allCovs = np.zeros((xdgmm.n_components, ndim, ndim))
        for gg in range(xdgmm.n_components):

            newMean, newCov, newAmp = multiplyGaussians(xdgmm.mu[gg], xdgmm.V[gg], mean2[dimension], cov2[dimension])
            newAmp *= xdgmm.weights[gg]
            allMeans[gg] = newMean
            allAmps[gg] = newAmp
            allCovs[gg] = newCov

            summedPosterior[k,:] += st.gaussian(newMean[projectedDimension], np.sqrt(newCov[projectedDimension, projectedDimension]), xabsMagKinda, amplitude=newAmp)
            individualPosterior[gg,:] = st.gaussian(newMean[projectedDimension], np.sqrt(newCov[projectedDimension, projectedDimension]), xabsMagKinda, amplitude=newAmp)

        for gg in range(xdgmm.n_components):
            points = drawEllipse.plotvector(allMeans[gg], allCovs[gg])
            axPost[0].plot(points[0, :], absMagKinda2absMag(points[1,:]), 'k', alpha=allAmps[gg]/np.max(allAmps), lw=0.5)
            axDist[3].plot(points[0, :], absMagKinda2absMag(points[1,:]), 'k', alpha=allAmps[gg]/np.max(allAmps), lw=0.5)
            points = drawEllipse.plotvector(xdgmm.mu[gg], xdgmm.V[gg])
            axPost[0].plot(points[0,:],absMagKinda2absMag(points[1,:]), 'r', lw=0.5, alpha=xdgmm.weights[gg]/np.max(xdgmm.weights))
            if k == 0:
                axDist[0].plot(points[0,:],absMagKinda2absMag(points[1,:]), 'r', lw=0.5, alpha=xdgmm.weights[gg]/np.max(xdgmm.weights))
                axDist[3].plot(points[0,:],absMagKinda2absMag(points[1,:]), 'r', lw=0.5, alpha=xdgmm.weights[gg]/np.max(xdgmm.weights))
        normalization = np.sum(allAmps)
        print 'the summed amplitudes are :', np.sum(allAmps)
        #if normalization != np.sum(allAmps):
        #    pdb.set_trace()
        summedPosterior[k,:] = summedPosterior[k,:]/normalization
        normalization_parallaxPosterior = scipy.integrate.cumtrapz(summedPosterior[k,:]*10.**(0.2*apparentMagnitude), xparallaxMAS)[-1]
        normalization_distancePosterior = scipy.integrate.cumtrapz(summedPosterior[k,:][positive]*xparallaxMAS[positive]**2.*10.**(0.2*apparentMagnitude), 1./xparallaxMAS[positive])[-1]

        normalization_logdistancePosterior = scipy.integrate.cumtrapz(summedPosterior[k,:][positive]*xparallaxMAS[positive]*10.**(0.2*apparentMagnitude)/np.log10(np.exp(1)), np.log10(1./xparallaxMAS[positive]))[-1]
        print 'the sum of parallax PDF is: ',normalization_parallaxPosterior
        print 'the sum of distance PDF is :', normalization_distancePosterior
        print 'the sum of log distance PDF is :', normalization_logdistancePosterior

        pointsData = drawEllipse.plotvector(mean2[dimension], cov2[dimension])

        axPost[0].plot(pointsData[0, :], absMagKinda2absMag(pointsData[1,:]), 'g')
        axDist[0].plot(pointsData[0, :], absMagKinda2absMag(pointsData[1,:]), 'g')
        axDist[3].plot(pointsData[0, :], absMagKinda2absMag(pointsData[1,:]), 'g', alpha=0.5)

        axPost[1].plot(xparallaxMAS, summedPosterior[k,:]*10.**(0.2*apparentMagnitude), 'k', lw=2)
        axPost[1].plot(xparallaxMAS, st.gaussian(data2[index], err2[index], xabsMagKinda)*10.**(0.2*apparentMagnitude), 'g', lw=2)
        axDist[1].plot(xparallaxMAS, st.gaussian(data2[index], err2[index], xabsMagKinda), 'g', lw=2, alpha=0.5)
        axDist[4].plot(xparallaxMAS, summedPosterior[k,:]*10.**(0.2*apparentMagnitude), 'k', lw=2, alpha=0.5)

        axPost[2].plot(1./xparallaxMAS[positive], summedPosterior[k,:][positive]*xparallaxMAS[positive]**2.*10.**(0.2*apparentMagnitude), 'k', lw=2)
        axPost[2].plot(1./xparallaxMAS[positive], st.gaussian(data2[index], err2[index], xabsMagKinda)[positive]*10.**(0.2*apparentMagnitude), 'g', lw=2)
        axDistLin[0].plot(1./xparallaxMAS[positive], st.gaussian(data2[index], err2[index], xabsMagKinda)[positive], 'g', lw=2, alpha=0.5)
        axDistLin[1].plot(1./xparallaxMAS[positive], summedPosterior[k,:][positive]*xparallaxMAS[positive]**2.*10.**(0.2*apparentMagnitude), 'k', lw=2, alpha=0.5)

        axPost[3].plot(np.log10(1./xparallaxMAS[positive]), summedPosterior[k,:][positive]*xparallaxMAS[positive]*10.**(0.2*apparentMagnitude)/np.log10(np.exp(1)), 'k', lw=2)
        axPost[3].plot(np.log10(1./xparallaxMAS[positive]), st.gaussian(data2[index], err2[index], xabsMagKinda)[positive]*10.**(0.2*apparentMagnitude), 'g', lw=2, label='likelihood')
        axDist[2].plot(np.log10(1./xparallaxMAS[positive]), st.gaussian(data2[index], err2[index], xabsMagKinda)[positive], 'g', lw=2, alpha=0.5)
        axDist[5].plot(np.log10(1./xparallaxMAS[positive]), summedPosterior[k,:][positive]*xparallaxMAS[positive]*10.**(0.2*apparentMagnitude)/np.log10(np.exp(1)), 'k', lw=2, alpha=0.5)


        axPost[0].set_xlim(-0.5, 2)
        axPost[0].set_ylim(9, -3)
        axPost[0].set_xlabel(xlabel, fontsize=18)
        axPost[0].set_ylabel(ylabel, fontsize=18)

        axPost[1].set_xlabel('Parallax [mas]', fontsize=18)
        axPost[1].set_xlim(-1, 6)
        axPost[1].set_ylabel(ylabel_posterior_parallax)

        #axPost[2].set_xscale('log')
        axPost[2].set_xlim(0.01, 3)
        axPost[2].set_xlabel('Distance [kpc]', fontsize=18)
        axPost[2].set_ylabel(ylabel_posterior_d)

        axPost[3].set_xlim(np.log10(0.3), np.log10(3))
        axPost[3].set_xlabel('log Distance [kpc]', fontsize=18)
        axPost[3].set_ylabel(ylabel_posterior_logd)
        plt.legend()
        plt.tight_layout()
        figPost.savefig('example.' + str(k) + '.png')
        np.save('summedPosteriorM67', summedPosterior)


    axDist[0].set_xlabel('B-V', fontsize=18)
    axDist[0].set_ylabel('M$_\mathrm{G}$', fontsize=18)
    axDist[0].set_xlim(-0.5, 2.0)
    axDist[0].set_ylim(9, -3)
    axDist[3].set_xlabel('B-V', fontsize=18)
    axDist[3].set_ylabel('M$_\mathrm{G}$', fontsize=18)
    axDist[3].set_xlim(-0.5, 2.0)
    axDist[3].set_ylim(9, -3)


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

    axDistLin[0].set_xlim(0, 2)
    axDistLin[1].set_xlim(0, 2)
    axDistLin[0].set_yscale('log')
    axDistLin[1].set_yscale('log')
    axDistLin[0].set_ylim(1e-4, 1e-2)
    axDistLin[1].set_ylim(1e-1, 50)
    axDistLin[0].set_xlabel('Distance [kpc]')
    axDistLin[1].set_xlabel('Distance [kpc]')
    axDistLin[0].set_ylabel(ylabel_likelihood_d)
    axDistLin[1].set_ylabel(ylabel_posterior_d)
    axDistLin[0].axvline(0.8, color='b', lw=2)
    axDistLin[0].axvline(0.9, color='b', lw=2)
    axDistLin[1].axvline(0.8, color='b', lw=2)
    axDistLin[1].axvline(0.9, color='b', lw=2)
    plt.tight_layout()

    figDistLin.savefig('distanceM67.linear.png')
