### score object
#"""
#
#A score class object 
# input:
#  data:[ [a,b,c], [],  ] (nx6)
#  entropy:[            ] (nx1)
#  kde:[                ] (nx1) if 1 is hard 0 easy
#  bin:[                ] (nx1) if 1 is hard 0 easy
#
#"""


# get weights
# correct/incorrect
# unique sorted set over weights
# bin-entropy-discrete-count
# continuous smooth sorted he indicator function
# hard-easy 
# alerts over global 000, 001, 010, 100, 101
# kernel-hash




#from math import fabs
from collections import Counter
#from entrofunc import *

import pickle, pprint
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from pylab import plot,show,hist,close
from matplotlib.pyplot import imshow

from astroML.plotting import hist
from astroML.density_estimation import bayesian_blocks

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

		#incorrect incorrect=1 correct=0
		incfile=open( incorrectpkl, 'rb' )
		incpkl = pickle.load(incfile)
		incnp = np.asarray( incpkl, dtype=np.float64)
		incfile.close()

		#members
		self.wts = wtnp
		self.entnrm = entnp
		self.inc = incnp 
		self.bins = None	
		self.entbys = None

	#def wei(self, self.wei):
	#	return self.wei

	def uniqueSrt(self):
	        c=Counter(); wht = np.copy(self.wts); arln = 1.0/4150
	        #dict key=weight : values=count-frequency
	        for e in wht:
	                c[e]+=1
			#interval count
			#ascending sort by key
			srt=np.unique( sorted(c) )
			return srt

	def bayesbin(self):
		#have an entropy over all the boosting estimators for 
		#over weights (N_readings x N_estimators)
		pass
	
	def entCntBin(self, *args ): 
	#return scalar, entropy for one reading
		c=Counter(); wht = np.copy(*args); arln = 1.0/4150
		#dict key=weight : values=count-frequency
		for e in wht:
			c[e]+=1
		
		#interval count
		srt=np.unique( sorted(c) ) #ascending sort by key
			
		#list iterator next()
		#i = iter(self.bins)
		i = [ x for x in np.nditer(self.bins)]
		dbn = i[0]

		#count number of val in binned interval
		frc=[] ; tvr=0; trx=0;
		#for x in np.nditer(self.bins, flags=['index']):
		for i in srt:
				if i < self.bins[trx]:
					tvr += c[i]
				else:
					frc.append(tvr)
					tvr=0
					trx+=1
					#dbn = i.next()
		
		#calculate entropy over binned counts
		etr = np.sum([ ((p * np.log2(p)) ,0.0) for p in frc if p>0.0])
        #print 'entropy ', etr
		return etr

	def entropy(self):
	#return for all readings
		entFeatures = self.wts
		c = Counter()
		for e in set(entFeatures.flat):      #get the unique  weight vals and number
			c[e]+=1
		entT = map(list, zip(*entFeatures))  #zip(*matrix) transposes the matrix row<=>col<br>
		ent = [self.entCntBin( e ) for e in entT  ]  #entCnt returns the count based entropy<br>
		return ent

	def pckle(self, selfdata, filename):
		pklfle = open(filename, 'wb')
		pickle.dump(selfdata,pklfle)
		pklfle.close
		
	def hardeasy(self):
		noInc = len(self.inc)
		total = len(self.entnrm) 
		noCrr = abs(total - noInc)
		print 'hardeasy', total, noInc, noCrr	
		
		#plot
		plt.figure(figsize(12.5, 4))
		colours = ["#348ABD", "#A60628"]

		hic = [100, noCrr-100] #hard_inc_corr
		eic = [noCrr-100, 300 ] #easy_inc_corr
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

	def entro(self, WT):
		#get the unique  weight vals and number
		c = Counter()
		for e in set(WT.flat):
			c[e]+=1
		#print 'T_counts ', c.most_common(10), ' ', sum( c.values() )

		#entCnt returns the count based entropy
		entT = map(list, zip(*entFeatures))  #map, dont use xrange #zip(*matrix) transposes row<=>col 
		ent = [entCnt( e ) for e in entT  ]

	def histo(self):
		#------------------------------------------------------------
		# First figure: show normal histogram binning
		#fig = plt.figure(figsize=(10, 4))
		fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15)

		ax1 = fig.add_subplot(121)
		ax1.hist(self.entnrm, bins=15, histtype='stepfilled', alpha=0.2, normed=True)
		ax1.set_xlabel('entropy bins=15')
		ax1.set_ylabel('Count(t)')

		ax2 = fig.add_subplot(122)
		ax2.hist(self.entnrm, bins=200, histtype='stepfilled', alpha=0.2, normed=True)
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
		    hist(self.entnrm, bins=200, histtype='stepfilled',
		         alpha=0.2, normed=True, label='standard histogram')
		
		    # plot an adaptive-width histogram on top
		    hist(self.entnrm, bins='blocks', ax=ax, color='black',
		         histtype='step', normed=True, label=title)
		
		    ax.legend(prop=dict(size=12))
		    ax.set_xlabel('entropy bins')
		    ax.set_ylabel('C(t)')
		
		plt.show()


	def comparePlot(self):
 		#hard vs easy
		#correct vs incorrect
		#correct vs incorrect
		pass

### main

# load data 
# workaround for module path issues
w = '/home/solver/project/scikit-learn-master/weight.pkl'
e = '/home/solver/project/scikit-learn-master/entropybasic.pkl'
i = '/home/solver/project/scikit-learn-master/incorrect.pkl'

ab = score(w,e,i)

# bin
import os.path
if os.path.isfile('bayesblock.pkl') :
		bayesfile=open('bayesblock.pkl','rb')
		bypkl = pickle.load(bayesfile)
		bynp = np.asarray( bypkl, dtype=np.float64 )
		ab.bins = bynp
		bayesfile.close()
else :
	ww = ab.wts.flat
	intervals =  bayesian_blocks(ww) #array of optimal bin_edges
	ab.bins = intervals
	ab.pckle(ab.bins,'bayesblock.pkl') 
	print 'bins**', ab.bins

# entropy
ent4150  = ab.entnrm #fixed-width 1/4150
entbayes = ab.entropy() #adaptive-width
#print 'entbayes', entbayes

### entropy
a = [[1,'z'],[5,'a'],[10,'b']]
import collections
vital = collections.namedtuple('vital',['ids','value'])
map(vital._make,a)
#vv=collections.Counter(dict(v2))
#vital._asdict()
#print vital
#3print vital.value
#sorted(vital, key=lambda v: v.value)
#print 'v', vital
#for ids, value in vital.iteritems(): #iteritems() not work
#	if value=='m':
#		print '*',ids

#weid = collections.namedtuple('id',
###########got to sleep
#use counter, sort, iteritems...its a dict

#cnt=[]
#def bincnt(i,j):
#	if j<i:
#		cnt+=1
#	else:
#		return cnt
#entCnt = [bincnt(i,j) for i in itrvl for j in ww]

#smooth
#http://jpktd.blogspot.com/2009/03/using-gaussian-kernel-density.html
#http://scikit-learn.org/stable/modules/density.html
#<a knowl="ref.html">kwl</a>
#Density estimation 

kernel = stats.gaussian_kde(ent4150)
hist(ent4150,normed=1,alpha=.3) # histogram over all points
print hist, kernel



### plot
#fig=plt.figure(figsize(10,4))

#plt.figure(figsize(12.5, 4))
colours = ["#348ABD", "#A60628"]

hic = [100, 100] #hard_inc_corr
eic = [200, 300 ] #easy_inc_corr
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
