### outline: score
# simpson paradox
# qq, cmdf, k-smirnov
# rank-sum, mann-whitney... p-val

### refs
#entropy
#https://www.ncbi.nlm.nih.gov/pubmed/10843903


"""

<script type="text/javascript" src="jquery-latest.min.js"></script>
<link href="knowlstyle.css" rel="stylesheet" type="text/css" />
<script type="text/javascript" src="knowl.js"></script>

"""


"class score"
"input:"
"  data:[ [a,b,c], [],  ] (nx6)"
"  entropy:[            ] (nx1)"
"  kde:[                ] (nx1) if 1 is hard 0 easy"
"  bin:[                ] (nx1) if 1 is hard 0 easy "





###
# hard-easy 
# alerts over global 000, 001, 010, 100, 101
# kernel-hash

from collections import Counter
import pickle, pprint
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from pylab import plot,show,hist,close
from matplotlib.pyplot import imshow

from astroML.plotting import hist
from astroML.density_estimation import bayesian_blocks
import pandas as pd

from scipy.stats import kde
from scipy.stats import norm
import matplotlib.mlab as mlab
 
#from pylab import *
from numpy import loadtxt
from scipy.optimize import leastsq
import csv
from itertools import izip
import os.path

import pandas as pd

class score(object):

### data
# init
	def __init__(self, weightpkl, entrpkl, incorrectpkl, datapkl):
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

		#data vital-headers, readings
		#dt = [ vitals, alerts, readings ]
		datafile=open( datapkl, 'rb' )
		dpkl = pickle.load(datafile)
		dv = np.asarray( dpkl[0] )
		da = np.asarray( dpkl[1], dtype=np.float64)
		dr = np.asarray( dpkl[2], dtype=np.float64)
		datafile.close()

		#class instance members
		self.wts = wtnp
		self.entnrm = entnp
		self.inc = incnp #full size 1=inc 0=corr
		self.bins = None	
 		self.entbys = None
		self.data=[]
		#self.data[0] = dv; self.data[1]= da; self.data[2]=dr
		self.data.append(dv); self.data.append(da); self.data.append(dr);
		self.df = None #panda dataframe

### utility function: data
	def pckle(self, selfdata, filename):
	  	pklfle = open(filename, 'wb')
		pickle.dump(selfdata,pklfle)
	 	pklfle.close

	def pcklcsv(self):
		with open('pandacsv.csv', "wb") as ofile:
			writeF = csv.writer(ofile, delimiter=',')
			#vitals alerts readings
			#
			for row in izip( self.data[2] , self.data[1] ):
				ll = self.data[2].shape[1]; 
				row2 = [i for i in row[0]]
				row2.append(row[1])
				writeF.writerow( row2 )
					

### pandas data frame
	def pnd(self):
		dtf = self.data #[vitals,alerts,readings]
		dataframe = pd.Series(dtf[2], index=dtf[0])		
		print 'DATA**', dataframe	
			
	
		print pd.__version__
		return dataframe

	def chckfile( fle, func ):
		#pickle
		if os.path.isfile( fle ) :
				bayesfile=open(fle , 'rb')
				bypkl = pickle.load(bayesfile)
				bynp = np.asarray( bypkl, dtype=np.float64 )
				ab.bins = bynp
				bayesfile.close()
		#function
		else :
			ww = ab.wts.flat
			intervals =  bayesian_blocks(ww) #array of optimal bin_edges
			ab.bins = intervals
			ab.pckle(ab.bins, fle ) 
			print 'bins**', ab.bins


	def dframe(self):
		"get scr attributes"
 		"[vitals,alerts,readings]"
		
		" open csv import "
		#df = pd.read_csv('pandacsv.csv')
		#header = self.data[0]
		#header = ['SYS','DIA','HR1','OX','HR2','WHT','alert']
		#df.columns = header
		
		"frame"
		hdr = [ i[0] for i in self.data[0] ]
		df2 = pd.DataFrame(self.data[2], columns=hdr[:6] ) #remove Label
		#print 'lbl', len(self.data[2][:,5])
		df2['Label']=self.data[2][:,5]
		
		
		"incorrect" "entropy" "weights"
		#df['INC'] = self.inc
		
		df2['ENT'] = self.entbys
		#print 'e len\n' , self.entbys[-5:-1],'\n', df2.tail(5)
		
		# 4150x50 df['WTS'] = self.wts

		"data frame attribute"
		self.df = df2
		print self.df.describe()


### entropy
# bayes binning

	def uniqueSrt(self):
	        c=Counter(); wht = np.copy(self.wts); arln = 1.0/4150
	        #dict key=weight : values=count-frequency
	        for e in wht:
	                c[e]+=1
			#interval count
			#ascending sort by key
			srt=np.unique( sorted(c) )
			return srt


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
		self.entbys = ent
	 	return ent

	



### histogram
# A histogram is discrete, count of number of values in a bin.   It is a frequency distribution of measurements, where the mean value can be seen directly(the highest bar).  Given a histogram the standard deviation, is the region centered at the mean value.  For a normal distribution, 68% of the measurements lie within one standard deviation on either side of the mean.  From a histogram, the width of the area which contains 68% of the measurements is 2stdv, or the width/2 for 1 stdv.  Histograms can be used to check theoretical distributions against observed data, comparing populatinos, or deriving particle velocity from time dilation (a proof of special relativity).  They answer the question of how many and where?  Viewing a histogram reveals whether the distribution is symmetric, left-skewed, or right-skewed, the number of modes, and any outliers.

# What is the probability that a sample taken from the signal will be within a certain state or bin? What is the probability a randomly chosen sample will have value greater than some number?  Integrating under the area of frequency counts (sum=1) in a histogram gives a probability distribution.  For example, a square have has a pdf with only two possible values, a triangle wave has a uniform distribution, and random noise has a gaussian (bell-shaped curve).  Thus, rather than having a frequency count of numbers in a bin, a probability of a number being in a bin can be given.  Likewise, a cumulative distribution function, gives the probability of finding some number, at any given time, in a range of bins, or standard deviations.  The gaussian distribution cannot be simply integrated, it requires sampling and tabulation of millions of points.  The P(x) would be restricted to those x-nubmer of states a signal takes.   

# The clt (central limit thm.) states that sum of random numbers becomes gaussian as more r.v. are summed.  The clt does not require the r.v to be from any particular distribution, or all from the same.  This is why normally distributed signal are seen widely in nature.  When many different random forces are interacting, the resulting pdf becomes Gaussian.   For example, adding one random number generates a step function, two rv generate uniform distribution(triangle wave), and many r.v generate a gaussian.  Often the number of states, or bin-sizes far exceeds number of samples in a signal.  Selecting bin number is a compromise of the statistical noise in two directions(x,y).  If too many bins, there are few counts, and the amplitude(y) is difficult to estimate.  If too few, the probability mass function (discrete), loses resolution along the x-axis.  

# robustness mean vs median
#http://stats.stackexchange.com/questions/59838/standard-error-of-the-median

	def ecdf(self, ent):
		plt.hist(ent, bins=self.bins, normed=True, cumulative=True)
		plt.title("cumulative distribution")
		plt.xlabel("entropy")
		plt.ylabel("Frequency")
		plt.show


### kde 
# A kernel is a type of probability distribution function, required to be even.  A kernel is non-negative, real-valued, even, integral=1.  PDFs which are kernels include uniform(-1,1), and standard normal distributions.
# KDE estimates the pdf of a continuous random variable without assumption about its underlying distribution, non-parametrically.  At every point, a kernel is created with the point at its center; therefore kernel is symmetric.  PDF is then estimated by adding all of the kernel functions and dividing by number of data(non-negative, and normalizes).

# The bandwidth of a kernel is the standard deviation.  Chosen small for large, tightly packed data, larger for sparse, small data sets.  


# kernel
#kernel = stats.gaussian_kde(ent4150)
#plt.title("kde")
#kdeX = np.linspace(0,5,100)
#plot(kdeX,kernel(kdeX),'r',label='kde') # distribution function
#print 'krnl', kernel
#hist(ent4150,normed=1,alpha=.3) # histogram over all points
	def plotkde(self):
		#kernel = stats.gaussian_kde( ent )
		#kdeX = np.linspace(0,5,100)
		#thresh = [ kernel(x) for x in kdeX ]
		#he = [   i for t in thresh ]
		
		#plot(kdeX,kernel(kdeX),'r',label='kde') # distribution function
 
		density = kde.gaussian_kde(self.entbys)
		xgrid = np.linspace(min(self.entbys), max(self.entbys), 1000)
		plt.hist(self.entbys, bins=8, normed=True)
		plt.plot(xgrid, density(xgrid), 'r-')
		plt.show()

	def plotkdefit(self):
		# best fit of data
		(mu, sigma) = norm.fit(self.entbys)
		
		# the histogram of the data
		n, bins, patches = plt.hist(self.entbys, 60, normed=1, facecolor='green', alpha=0.75)
		
		# add a 'best fit' line
		y = mlab.normpdf( bins, mu, sigma)
		l = plt.plot(bins, y, 'r--', linewidth=2)
		
		#plot
		plt.xlabel('Entropy')
		plt.ylabel('Probability')
		plt.title(r'$\mathrm{Histogram\ of\ Entropy:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))
		plt.grid(True)
		
		plt.show()	

	def plotkdeNL(self):
		fitfunc  = lambda p, x: p[0]*exp(-0.5*((x-p[1])/p[2])**2)+p[3]
		errfunc  = lambda p, x, y: (y - fitfunc(p, x))
		
		filename = "gaussdata.csv"
		data     = loadtxt(filename,skiprows=1,delimiter=',')
		xdata    = data[:,0]
		ydata    = data[:,1]
		
		init  = [1.0, 0.5, 0.5, 0.5]
		
		out   = leastsq( errfunc, init, args=(xdata, ydata))
		c = out[0]
		
		print "A exp[-0.5((x-mu)/sigma)^2] + k "
		print "Parent Coefficients:"
		print "1.000, 0.200, 0.300, 0.625"
		print "Fit Coefficients:"
		print c[0],c[1],abs(c[2]),c[3]
		
		plt.plot(xdata, fitfunc(c, xdata))
		plt.plot(xdata, ydata)
		
		plt.title(r'$A = %.3f\  \mu = %.3f\  \sigma = %.3f\ k = %.3f $' %(c[0],c[1],abs(c[2]),c[3]));
		
		plt.show()
	
	def ksggviolin(self, ent):
		pass


### summary statistics
# Error estimate can be obtained by repeating measurements in time.  The error, or deviations can be calculated as the difference from the mean/average over all measurements.  However, the average deviation over n samples, by definition, gives zero (the negative vs positive deviations cancel).  To remove the negative values, the erros are squared, and then averaged; this is the standard deviation measure.  The std tell us the average spread about the mean value.  Scales of precision for various measures: cm 0.1, electronic device half of last unit, 128.3 +- .015

# 5-number, box-plot

# Confidence intervals

# HDI vs tailed intervals
# using highest density interval instead of equal-tailed intervals
# Suppose heads=0 in n=10 flips of a coin, modeled with Bernoulli likelihood, and uniform prior dbeta(1,1).  Then the posterior = dbeta(1,11).  The graph of this curve is aysmptotic.  The 95% HDI includes zero, while the equal-tail intervals do not.  HDI is more meaningful and intuitive summary of the posterior.  
#http://doingbayesiandataanalysis.blogspot.com/2012/04/why-to-use-highest-density-intervals.html

# ECDF
# Cumulative density functions good way to distinguish distributions.  Present the data without transformation, unlike histograms or kde.  Give good visual representation, where accuracy of each point is buffered by points before and after without binning(histograms), or smoothing(kde).  Work well with parmaetric, mixture, or messy non-parametric distribution.  Shallow sloped areas represent sparse distribution, and steep represent dense distribution.
#http://stats.stackexchange.com/questions/51718/assessing-approximate-distribution-of-data-based-on-a-histogram/51753#51753 
#
#plot(ecdf(Annie),xlim=c(min(Zoe),max(Annie)),col="red",main="ECDFs")
#lines(ecdf(Brian),col="blue")
#lines(ecdf(Chris),col="green")
#lines(ecdf(Zoe),col="orange")
#


# qq, kir-smirnov
			

### threshold
#http://blog.counsyl.com/2013/08/07/detecting-genetic-copy-number-with-gaussian-mixture-models/

### signal processing
# time series
#- cross-validation
# http://www.dspguide.com/ch2/4.htm
# laplace transform
# z-transform
# benford law
 # kelly criterion



# apdx.
#	def rnkci(self): 
#
#		vitals = self.data[0]
#		alerts = self.data[1]
#		readings= self.data[2]
#		
#		
#		#heicg=[ "hard"=None, "easy"=None, "inc"=None, "cor"=None, "glb"=None ]
#		#return heicg 
#
#
#
#
#	def histo(self):
#		#------------------------------------------------------------
#		# First figure: show normal histogram binning
#		fig = plt.figure(figsize=(10, 4))
#		fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15)
#
#		ax1 = fig.add_subplot(121)
#		ax1.hist(self.entnrm, bins=15, histtype='stepfilled', alpha=0.2, normed=True)
#		ax1.set_xlabel('entropy bins=15')
#		ax1.set_ylabel('Count(t)')
#
#		ax2 = fig.add_subplot(122)
#		ax2.hist(self.entnrm, bins=200, histtype='stepfilled', alpha=0.2, normed=True)
#		ax2.set_xlabel('entropy bins=200')
#		ax2.set_ylabel('Count(t)')
#
#		#------------------------------------------------------------
#		# Second & Third figure: Knuth bins & Bayesian Blocks
#		fig = plt.figure(figsize=(10, 4))
#		fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15)
#		
#		for bins, title, subplot in zip(['knuth', 'blocks'],
#		                                ["Knuth's rule-fixed bin-width", 'Bayesian blocks variable width'],
#		                                [121, 122]):
#		    ax = fig.add_subplot(subplot)
#		
#		    # plot a standard histogram in the background, with alpha transparency
#		    hist(self.entnrm, bins=200, histtype='stepfilled',
#		         alpha=0.2, normed=True, label='standard histogram')
#		
#		    # plot an adaptive-width histogram on top
#		    hist(self.entbys, bins='blocks', ax=ax, color='black',
#		         histtype='step', normed=True, label=title)
#		
#		    ax.legend(prop=dict(size=12))
#		    ax.set_xlabel('entropy bins')
#		    ax.set_ylabel('C(t)')
#		
#		plt.show()
#
#
#	def comparePlot(self):
# 		#hard vs easy
#		#correct vs incorrect
#		#correct vs incorrect
#		pass
#
#
#
#
#def plot_accuracy(x, y, plot_placement, x_legend):
#    """Plot accuracy as a function of x."""
#    x = np.array(x)
#    y = np.array(y)
#    pl.subplots_adjust(hspace=0.5)
#    pl.subplot(plot_placement)
#    pl.title('Classification accuracy as a function of %s' % x_legend)
#    pl.xlabel('%s' % x_legend)
#    pl.ylabel('Accuracy')
#    pl.grid(True)
#    pl.plot(x, y)
#
#pl.figure(1)
#
## Plot accuracy evolution with #examples
#accuracy, n_examples = zip(*stats['accuracy_history'])
#plot_accuracy(n_examples, accuracy, 211, "training examples (#)")
#
## Plot accuracy evolution with runtime
#accuracy, runtime = zip(*stats['runtime_history'])
#plot_accuracy(runtime, accuracy, 212, 'runtime (s)')
#
#pl.show()
#
