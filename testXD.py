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

def convert2gal(ra, dec):
    return SkyCoord([ra, dec], unit=(units.hourangle, units.deg))

def m67indices(tgas, plot=False, dl=0.1, db=0.1):
    ra = '08:51:18.0'
    dec = '+11:49:00'
    l, b = '215.6960', '31.8963'
    index = (tgas['b'] < np.float(b) + db) & \
            (tgas['b'] > np.float(b) - db) & \
            (tgas['l'] < np.float(l) + dl) & \
            (tgas['l'] > np.float(l) - dl)
    if plot:
        plt.scatter(tgas['l'][index], tgas['b'][index], alpha=0.5, lw=0)
        plt.show()
    return index

def fixAbsMag(x):
    return 5.*np.log10(10.*x)

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

def multiplyGaussians(mean1, cov1, mean2, cov2):
    a = mean1
    b = mean2
    A = cov1
    B = cov2
    d = len(a)
    C = np.linalg.inv(np.linalg.inv(A) + np.linalg.inv(B))
    c = np.dot(np.dot(C,np.linalg.inv(A)),a) + np.dot(np.dot(C,np.linalg.inv(B)),b)
    exponent = -0.5*(np.dot(np.dot(np.transpose(a),np.linalg.inv(A)),a) + \
                     np.dot(np.dot(np.transpose(b),np.linalg.inv(B)),b) - \
                     np.dot(np.dot(np.transpose(c),np.linalg.inv(C)),c))
    z_c = (2*np.pi)**(-d/2.)*np.linalg.det(C)**0.5*np.linalg.det(A)**-0.5*np.linalg.det(B)**-0.5*np.exp(exponent)

    return c, C, z_c

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

    SNthreshold = 1
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

    indicesM67 = m67indices(tgasCutMatched, plot=True, db=1., dl=1.)

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

    fig, ax = plt.subplots()
    ax.scatter(apassCutMatched['bmag'] - apassCutMatched['vmag'], bayesDust)
    ax.set_xlabel('B-V', fontsize=20)
    ax.set_ylabel('E(B-V)', fontsize=20)
    #ax.set_ylim()
    #ax.set_yscale('log')
    fig.savefig('BV_vs_dust.png')


    temp = raveCutMatched['TEFF']/1000.
    temp_err = raveCutMatched['E_TEFF']/1000.

    absMagKinda = tgasCutMatched['parallax']*1e-3*10.**(0.2*tgasCutMatched['phot_g_mean_mag'])
    absMagKinda_err = tgasCutMatched['parallax_error']*1e-3*10.**(0.2*tgasCutMatched['phot_g_mean_mag'])
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
    #fig, axes = plt.subplots(1, 2, figsize=(12,5))
    #[np.array([[0.5, 6.], [1., 4.]]), np.array([[0.5, 1.], [1., 2.]])]


    for thresholdSN in [1]: #[16, 8, 4, 2, 1]:
    #for ngauss in [8, 128]:
        #thresholdSN = 1
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
            #mus = xdgmm.mu
            #Vs = xdgmm.V
            #amps = xdgmm.weights
            sample = xdgmm.sample(N)
            #dp.plot_sample(data1[j][indices], fixAbsMag(data2[j][indices]), data1[j][indices], fixAbsMag(data2[j][indices]),
            #       sample[:,0],fixAbsMag(sample[:,1]),xdgmm, xerr=err1[j][indices], yerr=fixAbsMag(err2[j][indices]), xlabel=xlabel[j], ylabel=r'M$_\mathrm{G}$')

            #os.rename('plot_sample.png', 'plot_sample_ngauss'+str(ngauss)+'.SN'+str(thresholdSN) + '.noSEED.png')
            badParallax = np.argsort(err2[j])[::-1]
            figtest, testax = plt.subplots(1, 2, figsize=(12,5))
            testax[1].scatter(sample[:,0],fixAbsMag(sample[:,1]), alpha=0.05, lw=0)
            for gg in range(xdgmm.n_components):
                points = drawEllipse.plotvector(xdgmm.mu[gg], xdgmm.V[gg])
                testax[0].plot(points[0,:],drawEllipse.fixAbsMag(points[1,:]), 'r', lw=0.5, alpha=xdgmm.weights[gg]/np.max(xdgmm.weights))

            for index in indicesM67: #np.random.randint(0, len(badParallax), 10)]:
                dimension = 0
                mean2, cov2 = matrixize(data1[j][index], data2[j][index], err1[j][index]**2., err2[j][index]**2.)
                print 'the data points mean are:', mean2, np.shape(mean2), ' the data points cov are:', cov2, np.shape(cov2)
                pointsData = drawEllipse.plotvector(mean2[dimension].T, cov2[dimension].T)
                #figtest, testax = plt.subplots(1, 2, figsize=(12,5))

                testax[1].plot(pointsData[0, :], drawEllipse.fixAbsMag(pointsData[1,:]), 'g-', alpha=1.0, lw=4)
                testax[0].plot(pointsData[0, :], drawEllipse.fixAbsMag(pointsData[1,:]), 'g-', alpha=1.0, lw=4)
                ndim = 2
                allMeans = np.zeros((xdgmm.n_components, ndim))
                allAmps = np.zeros(xdgmm.n_components)
                allCovs = np.zeros((xdgmm.n_components, ndim, ndim))
                for gg in range(xdgmm.n_components):
                    #points = drawEllipse.plotvector(xdgmm.mu[gg], xdgmm.V[gg])
                    #testax[0].plot(points[0,:],drawEllipse.fixAbsMag(points[1,:]), 'r', lw=0.5, alpha=xdgmm.weights[gg]/np.max(xdgmm.weights))
                    newMean, newCov, newAmp = multiplyGaussians(xdgmm.mu[gg], xdgmm.V[gg], mean2[dimension], cov2[dimension])
                    #print 'the new means are:', newMean, np.shape(newMean),' the new covs are:', newCov, np.shape(newCov), ' the new amps are:', newAmp, np.shape(newAmp)
                    allMeans[gg] = newMean
                    allAmps[gg] = newAmp
                    allCovs[gg] = newCov

                for gg in range(xdgmm.n_components):
                    points = drawEllipse.plotvector(allMeans[gg], allCovs[gg])
                    testax[1].plot(points[0, :], drawEllipse.fixAbsMag(points[1,:]), 'k-', lw=2, alpha=allAmps[gg]/np.max(allAmps)) #, alpha=newAmp/np.max(xdgmm.weights))
                    testax[0].plot(points[0, :], drawEllipse.fixAbsMag(points[1,:]), 'k-', lw=2, alpha=allAmps[gg]/np.max(allAmps))
                #print mean2, cov2, xdgmm.mu, xdgmm.V
            for j in [0,1]:
                testax[j].set_xlabel('B-V')
                testax[j].set_ylabel(r'M$_\mathrm{G}$')
            nsigma = 5.
                #print mean2[dimension][0], mean2[dimension][1], 3.*cov2[dimension][0,0], 3.*cov2[dimension][1,1]
            #testax[1].set_xlim(mean2[dimension][0] - nsigma*np.sqrt(cov2[dimension][0,0]), mean2[dimension][0] + nsigma*np.sqrt(cov2[dimension][0,0]))
            #testax[1].set_ylim(drawEllipse.fixAbsMag(mean2[dimension][1] + nsigma*np.sqrt(cov2[dimension][1,1])), drawEllipse.fixAbsMag(mean2[dimension][1] - nsigma*np.sqrt(cov2[dimension][1,1])))
            testax[0].set_xlim(-0.5, 2)
            testax[0].set_ylim(9, -3)
            #plt.tight_layout()
            plt.show()
            #figtest.savefig('posterior.png')
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
