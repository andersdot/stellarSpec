import numpy as np
import scipy.optimize as op

def make_fake_data(N=512):
    np.random.seed(42)
    mtrue, btrue, ttrue = -1.37, 0.2, 0.4
    xns = np.random.uniform(size=N) * 2. - 1.
    yntrues = mtrue * xns + btrue + np.random.normal(size=N) * ttrue
    sigmans = (np.random.uniform(size=N) * 1.5) ** 3
    yns = yntrues + np.random.normal(size=N) * sigmans
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

if __name__ == "__main__":
    import pylab as plt

    xns, yns, sigmans, yntrues = make_fake_data()
    plt.clf()
    plt.errorbar(xns, yns, yerr=sigmans, fmt="o", color="k", alpha=0.25)
    plt.errorbar(xns[0:16], yns[0:16], yerr=sigmans[0:16], fmt="o", color="k", zorder=37)
    plt.ylim(-5, 5)
    plt.savefig("data.png")

    m, b, t = get_best_fit_pars(xns, yns, sigmans)
    x1, x2 = plt.xlim()
    xp = np.array([x1, x2])
    plt.plot(xp, m * xp + b + t, "b-")
    plt.plot(xp, m * xp + b - t, "b-")
    plt.savefig("bestfit.png")

    ydns = np.zeros_like(yns)
    sigmadns = np.zeros_like(sigmans)
    for n, (xn, yn, sigman) in enumerate(zip(xns, yns, sigmans)):
        ydns[n], sigmadns[n] = denoise_one_datum(xn, yn, sigman, m, b, t)
    plt.errorbar(xns[0:16], ydns[0:16], yerr=sigmadns[0:16], fmt="o", color="b", zorder=37, alpha=0.75)
    plt.savefig("examples.png")

    plt.plot(xns[0:16], yntrues[0:16], "rx", ms=8., mew=2, zorder=39, alpha=0.75)
    plt.savefig("truths.png")

    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.clf()
    plt.errorbar(xns, ydns, yerr=sigmadns, fmt="o", color="b", alpha=0.25)
    plt.errorbar(xns[0:16], ydns[0:16], yerr=sigmadns[0:16], fmt="o", color="b", zorder=37, alpha=0.75)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.savefig("denoised.png")
