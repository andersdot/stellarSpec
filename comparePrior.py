import numpy as np
from xdgmm import XDGMM
import drawEllipse
import matplotlib.pyplot as plt
import matplotlib as mpl
import testXD
import sys

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



if __name__ == '__main__':
    #comparePrior()
    quantile = np.float(sys.argv[1])
    dustViz(quantile=quantile)
