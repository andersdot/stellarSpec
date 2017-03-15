import numpy as np
from xdgmm import XDGMM
import drawEllipse
import matplotlib.pyplot as plt
import testXD

def prior(xdgmm, ax):
    for gg in range(xdgmm.n_components):
        points = drawEllipse.plotvector(xdgmm.mu[gg], xdgmm.V[gg])
        ax[0].plot(points[0,:],testXD.absMagKinda2absMag(points[1,:]), c, lw=1, alpha=xdgmm.weights[gg]/np.max(xdgmm.weights))

if __name__ == '__main__':

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
        a.set_xlabel(r'$(J - K)$')
    ax[0].set_ylabel(r'$M_J$')
    ax[1].set_ylabel(r'$\varpi 10^{0.2\,m_J}$')
    ax[0].set_ylim(6, -6)
    ax[1].set_ylim(1100, -100)
    ax[1].legend(loc='lower left', fontsize=10)
    plt.tight_layout()
    fig.savefig('priorNgaussComparison.png')
