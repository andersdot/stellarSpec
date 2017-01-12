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

def timing(X, Xerr, nstars=1024, ngauss=64):
    amp_guess = np.zeros(ngauss) + np.random.rand(ngauss)
    cov_guess = np.zeros(((ngauss,) + X.shape[-1:] + X.shape[-1:]))
    cov_guess[:,diag,diag] = 1.0
    mean_guess = np.random.rand(ngauss,2)*10.
    start = time.time()
    ed(X, Xerr, amp_guess, mean_guess, cov_guess)
    end = time.time()
    return end-start, nstars

if __name__ == '__main__':
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
    filename = 'cutMatchedArrays.tgasApassSN1.npz'

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

    B_RedCoeff = 3.626
    V_RedCoeff = 2.742
    g_RedCoeff = 3.303
    r_RedCoeff = 2.285
    i_RedCoeff = 1.698
    bayesDust = st.dust(tgasCutMatched['l']*units.deg, tgasCutMatched['b']*units.deg, np.median(distCutMatched, axis=1)*units.pc)
    #M_V = apassCutMatched['vmag'] - V_RedCoeff*bayesDust - meanMuMatched
    g_r = apassCutMatched['gmag'] - g_RedCoeff*bayesDust - (apassCutMatched['rmag'] - r_RedCoeff*bayesDust)
    r_i = apassCutMatched['rmag'] - r_RedCoeff*bayesDust - (apassCutMatched['imag'] - i_RedCoeff*bayesDust)

    B_V = apassCutMatched['bmag'] - B_RedCoeff*bayesDust - (apassCutMatched['vmag'] - V_RedCoeff*bayesDust)
    B_V_err = np.sqrt(apassCutMatched['e_bmag']**2. + apassCutMatched['e_vmag']**2.)

    temp = raveCutMatched['TEFF']/1000.
    temp_err = raveCutMatched['E_TEFF']/1000.

    absMagKinda = tgasCutMatched['parallax']*1e-3*10.**(0.2*tgasCutMatched['phot_g_mean_mag'])
    absMagKinda_err = np.sqrt(tgasCutMatched['parallax_error']**2. + 0.3**2.)*1e-3*10.**(0.2*tgasCutMatched['phot_g_mean_mag'])
    #print absMagKinda_err
    #plt.scatter(B_V, absMagKinda, alpha=0.1, lw=0)
    #plt.show()
    data1 = [B_V, B_V]
    data2 = [temp, absMagKinda]
    err1 = [B_V_err, B_V_err]
    err2 = [temp_err, absMagKinda_err]
    xlabel = ['B-V', 'B-V']
    ylabel = ['Teff [kK]', r'$\varpi 10^{0.2*m_G}$']
    ngauss = 1028
    N = 120000
    #[np.array([[0.5, 6.], [1., 4.]]), np.array([[0.5, 1.], [1., 2.]])]
    xdgmm = XDGMM(method='Bovy')
    fig, axes = plt.subplots(figsize=(7,7))
    optimize = False
    subset = False
    timing = False
    nstar = '1.2M'
    #fig, axes = plt.subplots(1, 2, figsize=(12,5))
    for j, ax in zip([1],[axes]):
        try: 
            xdgmm = XDGMM(filename='xdgmm.'+ str(ngauss) + 'gauss.'+nstar+'.fit')
        except IOError:

            if subset:
                ind = np.random.randint(0, len(data1[j]), size=1024)
                X = np.vstack([data1[j][ind], data2[j][ind]]).T
                Xerr = np.zeros(X.shape + X.shape[-1:])
                diag = np.arange(X.shape[-1])
                Xerr[:, diag, diag] = np.vstack([err1[j][ind]**2., err2[j][ind]**2.]).T
            else:
                X = np.vstack([data1[j], data2[j]]).T
                Xerr = np.zeros(X.shape + X.shape[-1:])
                diag = np.arange(X.shape[-1])
                Xerr[:, diag, diag] = np.vstack([err1[j]**2., err2[j]**2.]).T
                if timing:
                    numData = [1024, 2048, 4096, 8192]
                    totTime = np.zeros(4)
                    for i, ns in enumerate(numData): 
                        totalTime, numStar = timing(X, Xerr, nstars=ns, ngauss=64)
                        print totalTime, numStar
                        totTime[i] = totalTime
                        plt.plot(numData, totTime)
                        plt.savefig('timing64Gaussians.png')

            if optimize:    
                param_range = np.array([256, 512, 1024, 2048, 4096, 8182]) #, 2, 4, 8, 16, 32, 64, 128]) #, 4, 8, 16])
                shuffle_split = ShuffleSplit(len(X), 16, test_size=0.3)
                train_scores, test_scores = validation_curve(xdgmm, X=X, y=Xerr, param_name='n_components', param_range=param_range, n_jobs=3, cv=shuffle_split, verbose=1)
                np.savez('xdgmm_scores.npz', train_scores=train_scores, test_scores=test_scores)
                train_scores_mean = np.mean(train_scores, axis=1)
                train_scores_std  = np.std(train_scores, axis=1)
                test_scores_mean = np.mean(test_scores, axis=1)
                test_scores_std = np.std(test_scores, axis=1)
                dp.plot_val_curve(param_range, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std)
                

            xdgmm.n_components = ngauss
            xdgmm = xdgmm.fit(X, Xerr)
            xdgmm.save_model('xdgmm.'+ str(ngauss) + 'gauss.'+nstar+'.fit')
    sample = xdgmm.sample(N)
    dp.plot_sample(data1[j], fixAbsMag(data2[j]), data1[j], fixAbsMag(data2[j]), sample[:,0],fixAbsMag(sample[:,1]), xdgmm, xlabel=xlabel[j], ylabel='M$_\mathrm{G}$', xerr=err1[j], yerr=fixAbsMag(err2[j]))
    #dp.plot_sample(data1[j], data2[j], data1[j], data2[j], sample, xdgmm, xlabel=xlabel[j], ylabel=ylabel[j], xerr=err1[j], yerr=err2[j])    
    os.rename('plot_sample.png', 'plot_sample_ngauss'+str(ngauss)+'.png')
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
