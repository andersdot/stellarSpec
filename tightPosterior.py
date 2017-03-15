import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib as mpl
from dustmaps.bayestar import BayestarQuery
from astropy.coordinates import SkyCoord
import astropy.units as units
import comparePrior

def distanceFilename(ngauss, quantile, iter, survey, dataFilename):
    return 'distanceQuantiles.' + str(ngauss) + 'gauss.dQ' + str(quantile) + '.' + iter + '.' + survey + '.' + dataFilename

def dustFilename(ngauss, quantile, iter, survey, dataFilename):
    return 'dustCorrection.'    + str(ngauss) + 'gauss.dQ' + str(quantile) + '.' + iter + '.' + survey + '.' + dataFilename

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

    return tgas, twoMass, Apass, indices

ngauss = 128
thresholdSN = 0.001
survey = '2MASS'
dataFilename = 'All.npz'
survey = '2MASS'
quantile = 0.05
dataFilename = 'All.npz'
norm = mpl.colors.Normalize(vmin=-1, vmax=5)

iteration = ['8th']
color = ['black', 'purple', 'blue', 'green', 'yellow', 'orange', 'red']

    fig, ax = plt.subplots(2, 3, figsize=(12, 12))
    ax = ax.flatten()

    dustEBVnew = None
    if quantile == 0.5:
        dustFile = dustFilename(ngauss, 0.05, '5th', survey, dataFilename)
        dust = np.load(dustFile)
        dustEBV = dust['ebv']
    else:
        dustFile = dustFilename(ngauss, 0.05, '1st', survey, dataFilename)
        dust = np.load(dustFile)
        dustEBV = np.zeros(len(dust['ebv']))


print 'N stars matching all criteria: ', str(np.sum(indices))

tgas, twoMass, Apass, indices = dataArrays()
