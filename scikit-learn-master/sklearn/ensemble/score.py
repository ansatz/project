### score object
#"""
#
#A score class object 
#  data:[ [a,b,c], [],  ] (nx6)
#  entropy:[            ] (nx1)
#  kde:[                ] (nx1) if 1 is hard 0 easy
#  bin:[                ] (nx1) if 1 is hard 0 easy
#
#"""
#from math import fabs
#from collections import Counter
#from entrofunc import *

import pickle, pprint
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

from astroML.plotting import hist

# First figure: show normal histogram binning
fig = plt.figure(figsize=(10, 4))
fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15)

ax1 = fig.add_subplot(121)
ax1.hist([10,11,12], bins=15, histtype='stepfilled', alpha=0.2, normed=True)
ax1.set_xlabel('entropy bins=15')
ax1.set_ylabel('Count(t)')

ax2 = fig.add_subplot(122)
ax2.hist([10,11,12], bins=200, histtype='stepfilled', alpha=0.2, normed=True)
ax2.set_xlabel('entropy bins=200')
ax2.set_ylabel('Count(t)')

###
class score(object):
	def __init__(self, weightpkl, entrpkl, incorrectpkl):
		#entropy
		entfile=open(entrpkl,'rb')
		entpkl = pickle.load(entfile)
		entnp = np.asarray( entpkl, dtype=np.float64 )
		entfile.close()
	
		#weight
		wtfile=open(weightpkl, 'rb')
		wtpkl = pickle.load(wtfile)
		wtnp = np.asarray( wtpkl, dtype=np.float64 )
		wtfile.close()
	
		#incorrect
		incfile=open( incorrectpkl, 'rb' )
		incpkl = pickle.load(incfile)
		incnp = np.asarray( incpkl, dtype=np.float64)
		incfile.close()
	
		#members
		self.entropy = entnp
		self.wts = wtnp
		self.inc = incnp
	
	def entro(self, WT):
		#get the unique  weight vals and number
		c = Counter()
		for e in set(WT.flat):
			c[e]+=1
		print 'T_counts ', c.most_common(10), ' ', sum( c.values() )
	
		#entCnt returns the count based entropy
		entT = map(list, zip(*entFeatures))  #map, dont use xrange #zip(*matrix) transposes row<=>col 
		ent = [entCnt( e ) for e in entT  ]
	
	def histo(self):
		#------------------------------------------------------------
		# First figure: show normal histogram binning
		fig = plt.figure(figsize=(10, 4))
		fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15)
	
		ax1 = fig.add_subplot(121)
		ax1.hist(self.entropy, bins=15, histtype='stepfilled', alpha=0.2, normed=True)
		ax1.set_xlabel('entropy bins=15')
		ax1.set_ylabel('Count(t)')
	
		ax2 = fig.add_subplot(122)
		ax2.hist(self.entropy, bins=200, histtype='stepfilled', alpha=0.2, normed=True)
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
		    hist(self.entropy, bins=200, histtype='stepfilled',
		         alpha=0.2, normed=True, label='standard histogram')
		
		    # plot an adaptive-width histogram on top
		    hist(self.entropy, bins='blocks', ax=ax, color='black',
		         histtype='step', normed=True, label=title)
		
		    ax.legend(prop=dict(size=12))
		    ax.set_xlabel('entropy bins')
		    ax.set_ylabel('C(t)')
		
		plt.show()
	
	def smooth(self): pass
	
	#hard-easy-incorrect-correct
	def heic(self):
		noInc = len(self.inc)
		total = len(self.entropy) 
		noCrr = abs(total - noInc)
		print total, noInc, noCrr	
		hic = [100, noCrr-100] #hard_inc_corr
		eic = [noCrr-100, 300 ] #easy_inc_corr
	
	def comparePlot(self):
		#hard vs easy
		#correct vs incorrect
		#correct vs incorrect
		pass
	
### main
w = '/home/solver/project/scikit-learn-master/weight.pkl'
e = '/home/solver/project/scikit-learn-master/entropybasic.pkl'
i = '/home/solver/project/scikit-learn-master/incorrect.pkl'

ab = score(w,e,i)
ab.heic()

#plt.figure(figsize(10,4))
#plot


### rnkcrv
# bar plot hard-easy
colours = ["#348ABD", "#A60628"]
prior = [0.20, 0.80]
posterior = [1./3, 2./3]
plt.bar([0, .7], prior, alpha=0.70, width=0.25,
				 color=colours[0], label="hard",
				 lw="3", edgecolor=colours[0])
	
plt.bar([0+0.25, .7+0.25], posterior, alpha=0.7,
				        width=0.25, color=colours[1],
						        label="easy",
								        lw="3", edgecolor=colours[1])
	
plt.xticks([0.20, .95], ["Incorrect", "Correct"])
plt.title("Hard vs Easy pts for Incorrect/Correct Alert")
plt.ylabel("Number Alerts")
plt.legend(loc="upper left");
		
plt.show()





### state of mind  ecdf, subjective priors, probability as opinion summary 
