import matplotlib
#matplotlib.use('pdf')
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
    return SkyCoord([ra, dec], unit=(units.hourangle, units.deg))

def m67indices(tgas, plot=False, dl=0.1, db=0.1):
    ra = '08:51:18.0'
    dec = '+11:49:00'
    l, b = '215.6960', '31.8963' #M67 800 pc
    #l, b = '166.5707', '-23.5212' #pleiades 132 pc
    index = (tgas['b'] < np.float(b) + db) & \
            (tgas['b'] > np.float(b) - db) & \
            (tgas['l'] < np.float(l) + dl) & \
            (tgas['l'] > np.float(l) - dl)
    if plot:
        plt.scatter(tgas['l'][index], tgas['b'][index], alpha=0.5, lw=0)
        plt.savefig('clusterOnSky.png')
    return index

def absMagKinda2Parallax(absMagKinda, apparentMag):
    return absMagKinda/10.**(0.2*apparentMag)


def parallax2absMagKinda(parallaxMAS, apparentMag):
    return parallaxMAS*10.**(0.2*apparentMag)

def plotDistance(ax, mean, sigma, amp, apparentMag, logminParallax=-4., logmaxParallax=0., lw=1, color='black', alpha=1.0, npoints=10000):
    x = np.logspace(logminParallax, logmaxParallax, npoints)
    distance = 1./absMag2Parallax(x, apparentMag)
    varpiDist = st.gaussian(mean, sigma, x, amplitude=amp)
    ax.plot(distance, varpiDist, 'k', alpha=alpha, lw=lw, color=color)


def absMagKinda2absMag(absMagKinda):
    absMagKinda_in_arcseconds = absMagKinda/1e3
    return 5.*np.log10(10.*absMagKinda_in_arcseconds)

def XD(data_1, err_1, data_2, err_2, ngauss=2, mean_guess=np.array([[0.5, 6.], [1., 4.]]), w=0.0):

    amp_guess = np.zeros(ngauss) + 1.
    ndim = 2
    X = np.vstack([data_1, data_2]).T
    Xerr = np.zeros(X.shape + X.shape[-1:])
    diag = np.arange(X.shape[-1])
    Xerr[:,diag,diag] = np.vstack([err_1**2., err_2**2.]).T

    cov_guess = np.zeros(((ngauss,) + X.shape[-1:] + X.shape[-1:]))
    cov_guess[:,diag,diag] = 1.0 #np.random.rand(ndim, ngauss)
    #pdb.set_trace()
    ed(X, Xerr, amp_guess, mean_guess, cov_guess, w=w)

    return amp_guess, mean_guess, cov_guess

def scalability(numberOfStars = [1024, 2048, 4096, 8192]):
    totTime = np.zeros(4)
    for i, ns in enumerate(numberOfStars):
        totalTime, numStar = timing(X, Xerr, nstars=ns, ngauss=64)
        print totalTime, numStar
        totTime[i] = totalTime
        plt.plot(numData, totTime)
        plt.savefig('timing64Gaussians.png')


def timing(X, Xerr, nstars=1024, ngauss=64):
    amp_guess = np.zeros(ngauss) + np.random.rand(ngauss)
    cov_guess = np.zeros(((ngauss,) + X.shape[-1:] + X.shape[-1:]))
    cov_guess[:,diag,diag] = 1.0
    mean_guess = np.random.rand(ngauss,2)*10.
    start = time.time()
    ed(X, Xerr, amp_guess, mean_guess, cov_guess)
    end = time.time()
    return end-start, nstars

def subset(data1, data2, err1, err2, nsamples=1024):
    ind = np.random.randint(0, len(data1[j]), size=nsamples)
    return matrixize(data1[ind], data2[ind], err1[ind], err2[ind])

def matrixize(data1, data2, err1, err2):
    X = np.vstack([data1, data2]).T
    Xerr = np.zeros(X.shape + X.shape[-1:])
    diag = np.arange(X.shape[-1])
    Xerr[:, diag, diag] = np.vstack([err1**2., err2**2.]).T
    return X, Xerr

def optimize(param_range=np.array([256, 512, 1024, 2048, 4096, 8182])):
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
    minabsMagKinda = parallax2absMagKinda(minParallaxMAS, apparentMagnitude)
    maxabsMagKinda = parallax2absMagKinda(maxParallaxMAS, apparentMagnitude)
    xabsMagKinda = np.linspace(minabsMagKinda, maxabsMagKinda, nPosteriorPoints)
    xparallaxMAS = np.linspace(minParallaxMAS, maxParallaxMAS, nPosteriorPoints)
    return xparallaxMAS, xabsMagKinda


if __name__ == '__main__':
    np.random.seed(2)
    b_v_lim = [0.25, 1.5]
    g_r_lim = None #[0, 1.5]

    r_i_lim = None #[-0.25, 0.75]
    M_v_lim = None #[10, 2]

    teff_lim = [7, 4] #kKd
    log_g_lim = [6, 3]
    feh_lim = [-1.5, 1]

    maxlogg = 20
    minlogg = 1
    mintemp = 100

    SNthreshold = 0.001
    filename = 'cutMatchedArrays.tgasApassSN0.npz'

    try:
        cutMatchedArrays = np.load(filename)
        tgasCutMatched = cutMatchedArrays['tgasCutMatched']
        apassCutMatched = cutMatchedArrays['apassCutMatched']
        raveCutMatched = cutMatchedArrays['raveCutMatched']
        twoMassCutMatched = cutMatchedArrays['twoMassCutMatched']
        wiseCutMatched = cutMatchedArrays['wiseCutMatched']
        distCutMatched = cutMatchedArrays['distCutMatched']
    except IOError:
        tgasCutMatched, apassCutMatched, raveCutMatched, twoMassCutMatched, wiseCutMatched, distCutMatched = st.observationsCutMatched(maxlogg=maxlogg, minlogg=minlogg, mintemp=mintemp, SNthreshold=SNthreshold, filename=filename)
    print 'Number of Matched stars is: ', len(tgasCutMatched)


    indicesM67 = m67indices(tgasCutMatched, plot=False, db=0.5, dl=0.5)


    B_RedCoeff = 3.626
    V_RedCoeff = 2.742
    g_RedCoeff = 3.303
    r_RedCoeff = 2.285
    i_RedCoeff = 1.698
    bayesDust = st.dust(tgasCutMatched['l']*units.deg, tgasCutMatched['b']*units.deg, np.median(distCutMatched, axis=1)*units.pc)
    #M_V = apassCutMatched['vmag'] - V_RedCoeff*bayesDust - meanMuMatched
    #g_r = apassCutMatched['gmag'] - g_RedCoeff*bayesDust - (apassCutMatched['rmag'] - r_RedCoeff*bayesDust)
    #r_i = apassCutMatched['rmag'] - r_RedCoeff*bayesDust - (apassCutMatched['imag'] - i_RedCoeff*bayesDust)
    #B_V = apassCutMatched['bmag'] - B_RedCoeff*bayesDust - (apassCutMatched['vmag'] - V_RedCoeff*bayesDust)

    #M_V = apassCutMatched['vmag'] - meanMuMatched
    g_r = apassCutMatched['gmag'] - apassCutMatched['rmag']
    r_i = apassCutMatched['rmag'] - apassCutMatched['imag']
    B_V = apassCutMatched['bmag'] - apassCutMatched['vmag']
    B_V_err = np.sqrt(apassCutMatched['e_bmag']**2. + apassCutMatched['e_vmag']**2.)
    #plt.hist(B_V_err, bins=100)
    #plt.show()
    """
    fig, ax = plt.subplots()
    ax.scatter(apassCutMatched['bmag'] - apassCutMatched['vmag'], bayesDust)
    ax.set_xlabel('B-V', fontsize=20)
    ax.set_ylabel('E(B-V)', fontsize=20)
    #ax.set_ylim()
    #ax.set_yscale('log')
    fig.savefig('BV_vs_dust.png')
    """

    temp = raveCutMatched['TEFF']/1000.
    temp_err = raveCutMatched['E_TEFF']/1000.

    absMagKinda = tgasCutMatched['parallax']*10.**(0.2*tgasCutMatched['phot_g_mean_mag'])
    absMagKinda_err = tgasCutMatched['parallax_error']*10.**(0.2*tgasCutMatched['phot_g_mean_mag'])
    #absMagKinda_err = np.sqrt(tgasCutMatched['parallax_error']**2. + 0.3**2.)*1e-3*10.**(0.2*tgasCutMatched['phot_g_mean_mag'])

    data1 = [B_V, B_V]
    data2 = [temp, absMagKinda]
    err1 = [B_V_err, B_V_err]
    err2 = [temp_err, absMagKinda_err]
    xlabel = ['B-V', 'B-V']
    ylabel = ['Teff [kK]', r'$\varpi 10^{0.2*m_G}$']
    ngauss = 128
    N = 120000
    optimize = False
    subset = False
    timing = False
    nstar = '1.2M'

    thresholdSN = 0.001


    fig, axes = plt.subplots(figsize=(7,7))
    parallaxSNcut = tgasCutMatched['parallax']/tgasCutMatched['parallax_error'] >= thresholdSN
    sigMax = 1.086/thresholdSN
    lowPhotErrorcut = (apassCutMatched['e_bmag'] < sigMax) & (apassCutMatched['e_vmag'] < sigMax) & (apassCutMatched['e_gmag'] < sigMax) & (apassCutMatched['e_rmag'] < sigMax) & (apassCutMatched['e_imag'] < sigMax)

    indices = parallaxSNcut & lowPhotErrorcut
        #filename = 'xdgmm.1028gauss.1.2M.fit'
    filename = 'xdgmm.'+ str(ngauss) + 'gauss.'+nstar+ '.SN' + str(thresholdSN) + '.fit'
    for j, ax in zip([1],[axes]):
        try:
            xdgmm = XDGMM(filename=filename)
        except IOError:
            if subset: X, Xerr = subset(data1[j][indices], data2[j][indices], err1[j][indices], err2[j][indieces], nsamples=1024)
            else: X, Xerr = matrixize(data1[j][indices], data2[j][indices], err1[j][indices], err2[j][indices])

            if timing: scalability(numberOfStars=[1024, 2048, 4096, 8192])
            if optimize:
                    test_scores, train_scores = optimize(param_range=np.array([256, 512, 1024, 2048, 4096, 8182]))
                    #maybe pick ngauss based on train_scores vs test_scores?


            try:
                xdgmm = XDGMM(method='Bovy', mu=mus, V=Vs, weights=amps)
            except NameError:
                print 'This is the first fit'
                xdgmm = XDGMM(method='Bovy')
            xdgmm.n_components = ngauss
            xdgmm = xdgmm.fit(X, Xerr)
            xdgmm.save_model(filename)
        sample = xdgmm.sample(N)
        figPrior, axPrior = plt.subplots()
        for gg in range(xdgmm.n_components):
            points = drawEllipse.plotvector(xdgmm.mu[gg], xdgmm.V[gg])
            axPrior.plot(points[0,:],absMagKinda2absMag(points[1,:]), 'r', lw=0.5, alpha=xdgmm.weights[gg]/np.max(xdgmm.weights))
        axPrior.invert_yaxis()
        figPrior.savefig('prior.png')

        #dp.plot_sample(data1[j][indices], fixAbsMag(data2[j][indices]), data1[j][indices], fixAbsMag(data2[j][indices]),
        #       sample[:,0],fixAbsMag(sample[:,1]),xdgmm, xerr=err1[j][indices], yerr=fixAbsMag(err2[j][indices]), xlabel=xlabel[j], ylabel=r'M$_\mathrm{G}$')

        #os.rename('plot_sample.png', 'plot_sample_ngauss'+str(ngauss)+'.SN'+str(thresholdSN) + '.noSEED.png')

        nPosteriorPoints = 10000
        summedPosterior = np.zeros((np.sum(indicesM67), nPosteriorPoints))
        projectedDimension = 1
        figDist, axDist = plt.subplots(2, 2, figsize=(15, 15))
        axDist = axDist.flatten()
        for k, index in enumerate(np.where(indicesM67)[0]):
            individualPosterior = np.zeros((xdgmm.n_components, nPosteriorPoints))
            figPost, axPost = plt.subplots(1, 3, figsize=(17,7))
            figtest, testax = plt.subplots(1, 3, figsize=(17,7))
            windowFactor = 5.
            minParallaxMAS = tgasCutMatched['parallax'][index] - windowFactor*tgasCutMatched['parallax_error'][index]
            maxParallaxMAS = tgasCutMatched['parallax'][index] + windowFactor*tgasCutMatched['parallax_error'][index]
            apparentMagnitude = tgasCutMatched['phot_g_mean_mag'][index]
            xparallaxMAS, xabsMagKinda = plotXarrays(minParallaxMAS, maxParallaxMAS, apparentMagnitude, nPosteriorPoints=nPosteriorPoints)
            positive = xparallaxMAS > 0.
            print 'the min and max of xparallax is: ', np.min(xparallaxMAS), np.max(xparallaxMAS)
            print 'the measured parallax is: ', tgasCutMatched['parallax'][index]

            dimension = 0
            mean2, cov2 = matrixize(data1[j][index], data2[j][index], err1[j][index], err2[j][index])
            pointsData = drawEllipse.plotvector(mean2[dimension], cov2[dimension])

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

                points = drawEllipse.plotvector(xdgmm.mu[gg], xdgmm.V[gg])
                axPost[0].plot(points[0,:],absMagKinda2absMag(points[1,:]), 'r', lw=0.5, alpha=xdgmm.weights[gg]/np.max(xdgmm.weights))

            normalization = scipy.integrate.cumtrapz(summedPosterior[k,:], xabsMagKinda)[-1]
            print 'the normalization is :', normalization
            print 'the summed amplitudes are :' np.sum(allAmps)

            summedPosterior[k,:] = summedPosterior[k,:]/normalization

            axPost[0].plot(pointsData[0, :], absMagKinda2absMag(pointsData[1,:]), 'g')

            axPost[1].plot(xparallax*1e3, summedPosterior[k,:], 'k', lw=2)
            axPost[1].plot(xparallax*1e3, st.gaussian(data2[j][index], err2[j][index], xabsMagKinda), 'g', lw=2)

            axPost[2].plot(1./xparallax[positive], summedPosterior[k,:][positive], 'k', lw=2)
            axPost[2].plot(1./xparallax[positive], st.gaussian(data2[j][index], err2[j][index], xabsMagKinda)[positive], 'g', lw=2)

            axPost[0].set_xlim(-0.5, 2)
            axPost[0].set_ylim(9, -3)
            axPost[0].set_xlabel('B-V', fontsize=18)
            axPost[0].set_ylabel(r'M$_\mathrm{G}$', fontsize=18)
            axPost[1].set_xlabel('Parallax [mas]', fontsize=18)
            axPost[1].set_xlim(-1, 6)
            axPost[2].set_xscale('log')
            axPost[2].set_xlim(30, 3000)
            axPost[2].set_xlabel('Distance [pc]', fontsize=18)

            figPost.savefig('example.' + str(k) + '.png')

            axDist[0].plot(xparallax*1e3, st.gaussian(data2[j][index], err2[j][index], xabsMagKinda), 'g', lw=2, alpha=0.5)
            axDist[2].plot(xparallax*1e3, summedPosterior[k,:], 'k', lw=2, alpha=0.5)

            axDist[1].plot(1./xparallax[positive], st.gaussian(data2[j][index], err2[j][index], xabsMagKinda)[positive], 'g', lw=2, alpha=0.5)
            axDist[3].plot(1./xparallax[positive], summedPosterior[k,:][positive], 'k', lw=2, alpha=0.5)

            np.save('summedPosteriorM67', summedPosterior)


    axDist[1].axvline(800, color="b", lw=2)
    axDist[1].axvline(900, color="b", lw=2)
    axDist[3].axvline(800, color="b", lw=2)
    axDist[3].axvline(900, color="b", lw=2)
    axDist[0].axvline(1./800*1e3, color="b", lw=2)
    axDist[0].axvline(1./900*1e3, color="b", lw=2)
    axDist[2].axvline(1./800*1e3, color="b", lw=2)
    axDist[2].axvline(1./900*1e3, color="b", lw=2)

    axDist[0].set_xlim(-1, 6)
    axDist[2].set_xlim(-1, 6)
    axDist[1].set_xlim(30,3000)
    axDist[3].set_xlim(30,3000)

    axDist[1].set_xscale('log')
    axDist[3].set_xscale('log')
    axDist[1].set_xlabel('Distance [pc]')
    axDist[3].set_xlabel('Distance [pc]')
    axDist[0].set_xlabel('Parallax [mas]')
    axDist[2].set_xlabel('Parallax [mas]')
    axDist[0].set_ylabel('Likelihood')
    axDist[2].set_ylabel('Posterior')
    figDist.savefig('distancesM67.png')

"""
        mean_guess = np.random.rand(ngauss,2)*10.
        X_train, X_test, y_train, y_test, xerr_train, xerr_test, yerr_train, yerr_test = train_test_split(data1[j], data2[j], err1[j], err2[j], test_size=0.4, random_state=0)

        start = time.time()
        amp, mean, cov = XD(data1[j], err1[j], data2[j], err2[j], ngauss=ngauss, mean_guess=mean_guess, w=0.001)
        end = time.time()
        #print np.shape(amp), np.shape(mean), np.shape(cov)
        print 'Time to run XD: ', end - start
        print 'Amplitudes are :', amp
        print 'Means are: ', mean
        print 'Covariances are: ', cov

        ax.scatter(data1[j], data2[j], alpha=0.1, lw=0)
        ax.errorbar(data1[j], data2[j], xerr=err1[j], yerr=err2[j], fmt="none", ecolor='black', zorder=0, lw=0.5, mew=0, alpha=0.1)
        for scale, alpha in zip([1, 2], [1.0, 0.5]):
            for i in range(ngauss):
                st.draw_ellipse(mean[i][0:2], cov[i][0:2], scales=[scale], ax=ax,
                     ec='k', fc="None", alpha=alpha*amp[i]/np.max(amp), zorder=99, lw=2)
        ax.set_xlabel(xlabel[j])
        ax.set_ylabel(ylabel[j])
    ax.set_title('eXtreme Deconvolution')
    plt.tight_layout()
    fig.savefig('testXD_' + str(ngauss) + '.png')
"""
