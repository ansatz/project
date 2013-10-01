# this file will read entropy.csv and calculate the adjusted bin histogram using astroML

import pickle
#with open('entropy.csv', 'rb') as f:
#	reader = csv.reader(f)
#	ent = [row for row in reader]

entfile=open('entropy.txt')
ent = pickle.load(entfile)
entfile.close()
ent[-10:]
print 'ent', ent

import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

from astroML.plotting import hist

# draw a set of variables
#np.random.seed(0)
#t = np.concatenate([stats.cauchy(-5, 1.8).rvs(500),
#                    stats.cauchy(-4, 0.8).rvs(2000),
#                    stats.cauchy(-1, 0.3).rvs(500),
#                    stats.cauchy(2, 0.8).rvs(1000),
#                    stats.cauchy(4, 1.5).rvs(500)])
#
## truncate values to a reasonable range
#t = t[(t > -15) & (t < 15)]

#------------------------------------------------------------
# First figure: show normal histogram binning
fig = plt.figure(figsize=(10, 4))
fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15)

ax1 = fig.add_subplot(121)
ax1.hist(ent, bins=15, histtype='stepfilled', alpha=0.2, normed=True)
ax1.set_xlabel('entropy bins=15')
ax1.set_ylabel('Count(t)')

ax2 = fig.add_subplot(122)
ax2.hist(ent, bins=200, histtype='stepfilled', alpha=0.2, normed=True)
ax2.set_xlabel('entropy bins=200')
ax2.set_ylabel('Count(t)')

#------------------------------------------------------------
# Second & Third figure: Knuth bins & Bayesian Blocks
fig = plt.figure(figsize=(10, 4))
fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15)

for bins, title, subplot in zip(['knuth', 'blocks'],
                                ["Knuth's rule", 'Bayesian blocks'],
                                [121, 122]):
    ax = fig.add_subplot(subplot)

    # plot a standard histogram in the background, with alpha transparency
    hist(ent, bins=200, histtype='stepfilled',
         alpha=0.2, normed=True, label='standard histogram')

    # plot an adaptive-width histogram on top
    hist(ent, bins=bins, ax=ax, color='black',
         histtype='step', normed=True, label=title)

    ax.legend(prop=dict(size=12))
    ax.set_xlabel('entropy bins=adaptive-width')
    ax.set_ylabel('C(t)')

plt.show()

