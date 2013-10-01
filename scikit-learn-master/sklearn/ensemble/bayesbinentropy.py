# this file will read entropy.csv and calculate the adjusted bin histogram using astroML

#from pickle import load 
#from pprint import pprint
import pickle, pprint
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

from astroML.plotting import hist
# read the entropy values from entropy.pkl
# with open('entropy.csv', 'rb') as f:
#	reader = csv.reader(f)
#	ent = [row for row in reader]

entfile=open('/home/solver/project/scikit-learn-master/sklearn/ensemble/entropy.pkl','rb')
entpkl = pickle.load(entfile)
#ent = np.load(entfile)
entnp = np.asarray( entpkl, dtype=np.float64)
#ent = [ 1.0 if val==0 else val for val in entnp ]
entfile.close()
ent2=entnp
#ent2 = [
pprint.pprint(ent2)


#------------------------------------------------------------
# First figure: show normal histogram binning
fig = plt.figure(figsize=(10, 4))
fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15)

ax1 = fig.add_subplot(121)
ax1.hist(ent2, bins=15, histtype='stepfilled', alpha=0.2, normed=True)
ax1.set_xlabel('entropy bins=15')
ax1.set_ylabel('Count(t)')

ax2 = fig.add_subplot(122)
ax2.hist(ent2, bins=200, histtype='stepfilled', alpha=0.2, normed=True)
ax2.set_xlabel('entropy bins=200')
ax2.set_ylabel('Count(t)')

#------------------------------------------------------------
# Second & Third figure: Knuth bins & Bayesian Blocks
fig = plt.figure(figsize=(10, 4))
fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15)

for bins, title, subplot in zip(['knuth', 'blocks'],
                                ["Knuth's rule-fixed bin-width", 'Bayesian blocks variable width'],
                                [121, 122]):
    ax = fig.add_subplot(subplot)

    # plot a standard histogram in the background, with alpha transparency
    hist(ent2, bins=200, histtype='stepfilled',
         alpha=0.2, normed=True, label='standard histogram')

    # plot an adaptive-width histogram on top
    hist(ent2, bins='blocks', ax=ax, color='black',
         histtype='step', normed=True, label=title)

    ax.legend(prop=dict(size=12))
    ax.set_xlabel('entropy bins')
    ax.set_ylabel('C(t)')

plt.show()

#-----------------------------------------------------------
# kde smooth






#-------------------------------------------------------------
# splom
