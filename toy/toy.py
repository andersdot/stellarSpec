import numpy as np
import scipy.optimize as op
import sys


def make_fake_data(parsTrue, N=512):
    np.random.seed(42)
    mtrue, btrue, ttrue = parsTrue
    xns = np.random.uniform(size=N) * 2. - 1.
    tns = np.random.normal(scale=ttrue, size=N)
    #tns = np.random.normal(size=N) * ttrue
    yntrues = mtrue * xns + btrue + tns
    sigmans = (np.random.uniform(size=N) * 2.) ** 3.
    #yns = yntrues + np.random.normal(size=N) * sigmans
    yns = yntrues + np.random.normal(scale=sigmans)
    fig, ax = plt.subplots(1,3)
    ax[0].scatter(xns, tns, s=2)
    ax[0].set_ylabel('sampled noise from ttrue gaussian')
    ax[1].scatter(xns, sigmans, s=2)
    ax[1].set_ylabel('sigmans')
    ax[2].scatter(xns, np.random.normal(scale=sigmans), s=2)
    ax[2].set_ylabel('sampled noise from sigmans gaussian')
    plt.tight_layout()
    fig.savefig('laurenUnderstanding.png')
    return xns, yns, sigmans, yntrues

def objective(pars, xns, yns, sigmas):
    m, b, t = pars
    resids = yns - (m * xns + b)
    return np.sum(resids * resids / (sigmas * sigmas + t * t)
                  + np.log(sigmas * sigmas + t * t))

def get_best_fit_pars(xns, yns, sigmans):
    pars0 = np.array([-1.5, 0., 1.])
    result = op.minimize(objective, pars0, args=(xns, yns, sigmans))
    return result.x

def denoise_one_datum(xn, yn, sigman, m, b, t):
    s2inv = 1. / (sigman * sigman)
    t2inv = 1. / (t * t)
    return (yn * s2inv + (m * xn + b) * t2inv) / (s2inv + t2inv), \
        np.sqrt(1. / (s2inv + t2inv))

def plot(fig, axes, figNoise, axesNoise, xns, yns, sigmans, yntrues, m, b, t, mtrue, btrue, ttrue, ydns, sigmadns, nexamples=5):

    alpha_all = 0.05
    alpha_chosen = 1.0
    xlim = (-1, 1)
    ylim = (-5, 5)

    dataMap = mpl.cm.get_cmap('Blues')
    dataColor = dataMap(0.75)
    trueMap = mpl.cm.get_cmap('Reds')
    trueColor = trueMap(0.75)

    for ax in axes[:-1]:
        ax.errorbar(xns, yns, yerr=sigmans, fmt="o", color="k", alpha=alpha_all ,mew=0)
        ax.errorbar(xns[0:nexamples], yns[0:nexamples], yerr=sigmans[0:nexamples], fmt="o", color="k", zorder=37, alpha=alpha_chosen, mew=0)

    xp = np.array(xlim)
    axes[1].plot(xp, m * xp + b + t, color=dataColor)
    axes[1].plot(xp, m * xp + b - t, color=dataColor)
    axes[1].scatter(xns[0:nexamples], yntrues[0:nexamples], c=trueColor, lw=2, zorder=36, alpha=alpha_chosen, facecolors='None')
    axes[1].plot(xp, mtrue*xp + btrue + ttrue, color=trueColor, zorder=35)
    axes[1].plot(xp, mtrue*xp + btrue - ttrue, color=trueColor, zorder=34)
    r1 = axes[1].add_patch(mpl.patches.Rectangle((-10,-10), 0.1, 0.1, color='black', alpha=alpha_chosen))
    r2 = axes[1].add_patch(mpl.patches.Rectangle((-10,-10), 0.1, 0.1, color=trueColor, alpha=alpha_chosen))
    r3 = axes[1].add_patch(mpl.patches.Rectangle((-10,-10), 0.1, 0.1, color=dataColor, alpha=alpha_chosen))
    axes[1].legend((r1,r2,r3), ('data', 'truth', 'denoised'), loc='best', fontsize=12)

    axes[1].errorbar(xns[0:nexamples], ydns[0:nexamples], yerr=sigmadns[0:nexamples], fmt="o", color=dataColor, zorder=37, alpha=alpha_chosen, mew=0)

    norm = mpl.colors.Normalize(vmin=0, vmax=9)
    im = axes[2].scatter(xns,  ydns,  c=sigmans**2., cmap='Blues', norm=norm, alpha=0.5, lw=0)
    fig.subplots_adjust(left=0.05, right=0.89)
    cbar_ax = fig.add_axes([0.9, 0.125, 0.02, 0.75])
    cb = fig.colorbar(im, cax=cbar_ax)
    #cb = plt.colorbar(im, ax=axes[2])
    cb.set_label(r'$\sigma_n^2$', fontsize=20)
    cb.set_clim(-4, 9)


    axes[2].errorbar(xns, ydns, yerr=sigmadns, fmt="None", mew=0, color='black', alpha=0.25, elinewidth=0.5)
    #axes[2].errorbar(xns[0:nexamples], ydns[0:nexamples], yerr=sigmadns[0:nexamples], fmt="o", color="b", zorder=37, alpha=alpha_chosen, mew=0)
    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks((-1, -0.5, 0, 0.5, 1))
        ax.set_xticklabels((-1, 0.5, 0, 0.5, 1))
    #plt.tight_layout()

    axesNoise[0].hist(sigmans, bins=20, histtype='step', normed=True, lw=2)
    axesNoise[0].set_xlabel(r'$\sigma_n$', fontsize=15)
    axesNoise[1].hist(sigmadns, bins=20, histtype='step', normed=True, lw=2)
    axesNoise[1].set_xlabel(r'$\tilde\sigma_n$', fontsize=15)
    for a in axesNoise[0:2]:
        a.axvline(t, label='$t$', color='black', lw=2)
        a.legend(fontsize=15, loc='best')
    axesNoise[2].hist(yns-yntrues, bins=20, histtype='step', normed=True, lw=2)
    axesNoise[2].set_xlabel(r'$y_n - y_{true,n}$', fontsize=15)
    axesNoise[3].hist(ydns-yntrues, bins=20, histtype='step', normed=True, lw=2)
    axesNoise[3].set_xlabel(r'$<p(y_{true,n})> - y_{true,n}$', fontsize=15)
    #plt.tight_layout()

    return fig, axes, figNoise, axesNoise

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    nexamples =  np.int(sys.argv[1])
    mtrue, btrue, ttrue = -1.37, 0.2, 0.8
    parsTrue = [mtrue, btrue, ttrue] #m, b, t

    xns, yns, sigmans, yntrues = make_fake_data(parsTrue)
    m, b, t = get_best_fit_pars(xns, yns, sigmans)

    print m, b, t

    ydns = np.zeros_like(yns)
    sigmadns = np.zeros_like(sigmans)
    for n, (xn, yn, sigman) in enumerate(zip(xns, yns, sigmans)):
        ydns[n], sigmadns[n] = denoise_one_datum(xn, yn, sigman, m, b, t)

    for label, style in zip(['paper', 'talk'],['seaborn-paper', 'seaborn-talk']):

        plt.style.use(style)
        mpl.rcParams['xtick.labelsize'] = 18
        mpl.rcParams['ytick.labelsize'] = 18
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes = axes.flatten()
        figNoise, axesNoise = plt.subplots(2,2, figsize=(8,8))
        axesNoise = axesNoise.flatten()

        fig, axes, figNoise, axesNoise = plot(fig, axes, figNoise, axesNoise, xns, yns, sigmans, yntrues, m, b, t, mtrue, btrue, ttrue, ydns, sigmadns, nexamples=nexamples)
        figNoise.tight_layout()
        figNoise.savefig('toyNoise.' + label + '.png')
        fig.savefig("toy." + label + ".png")
