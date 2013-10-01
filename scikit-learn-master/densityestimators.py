#http://www.astroml.org/book_figures/chapter6/fig_density_estimation.html#book-fig-chapter6-fig-density-estimation
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from astroML.density_estimation import KDE, KNeighborsDensity
from astroML.plotting import hist

import pickle, pprint

#------------------------------------------------------------
entfile=open('/home/solver/project/scikit-learn-master/sklearn/ensemble/entropy.pkl','rb')
entpkl = pickle.load(entfile)
entnp = np.asarray( entpkl, dtype=np.float64)
entfile.close()
ent2=entnp
pprint.pprint(ent2)

#------------------------------------------------------------
# plot the results
fig = plt.figure(figsize=(8, 8))
fig.subplots_adjust()
N_values = (500, 5000)
subplots = (211, 212)
k_values = (10, 100)

for N, k, subplot in zip(N_values, k_values, subplots):
    ax = fig.add_subplot(subplot)
    xN = ent2[:N]
    t = np.linspace(-10, 30, 1000)

    # Compute density with KDE
    kde = KDE('gaussian', h=0.1).fit(xN[:, None])
    dens_kde = kde.eval(t[:, None]) / N

    # Compute density with Bayesian nearest neighbors
    nbrs = KNeighborsDensity('bayesian', n_neighbors=k).fit(xN[:, None])
    dens_nbrs = nbrs.eval(t[:, None]) / N

    # plot the results
    #ax.plot(t, true_pdf(t), ':', color='black', zorder=3,
    #        label="Generating Distribution")
    ax.plot(xN, -0.005 * np.ones(len(xN)), '|k', lw=1.5)
    hist(xN, bins='blocks', ax=ax, normed=True, zorder=1,
         histtype='stepfilled', lw=1.5, color='k', alpha=0.2,
         label="Bayesian Blocks")
    ax.plot(t, dens_nbrs, '-', lw=2, color='gray', zorder=2,
            label="Nearest Neighbors (k=%i)" % k)
    ax.plot(t, dens_kde, '-', color='black', zorder=3,
            label="Kernel Density (h=0.1)")

    # label the plot
    ax.text(0.02, 0.95, "%i points" % N, ha='left', va='top',
            transform=ax.transAxes)
    ax.set_ylabel('$p(x)$')
    ax.legend(loc='upper right', prop=dict(size=12))

    if subplot == 212:
        ax.set_xlabel('$x$')

    #ax.set_xlim(0, 500)
    #ax.set_ylim(-0.01, 0.4001)

plt.show()
