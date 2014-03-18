### outline: score
# simpson paradox
# qq, cmdf, k-smirnov
# rank-sum, mann-whitney... p-val

### refs
#entropy
#https://www.ncbi.nlm.nih.gov/pubmed/10843903
#http://stanford.edu/~mwaskom/software/seaborn/

#cross-validation
#http://bytefish.de/blog/gender_classification/
#http://normaldeviate.wordpress.com/
#bonferonni corrrected confidence interval
#time of day effect http://nbviewer.ipython.org/urls/raw.github.com/jrjohansson/scientific-python-lectures/master/Lecture-4-Matplotlib.ipynb
#http://blog.nextgenetics.net/?e=94

#plot
#http://nbviewer.ipython.org/urls/raw.github.com/jrjohansson/scientific-python-lectures/master/Lecture-4-Matplotlib.ipynb
#https://github.com/mikedewar/d3py
#https://github.com/wrobstory/vincent
#http://blog.nextgenetics.net/?e=85
#http://pandas.pydata.org/pandas-docs/dev/visualization.html#scatter-plot-matrix

#pandas
#http://nbviewer.ipython.org/urls/gist.github.com/wesm/5773719/raw/1399562c0a02b9edc3d13c71a70387a31d87260b/tutorial.ipynb
#www.bearrelroll.com/2013/05/python-pandas-tutorial/
#http://stackoverflow.com/questions/18598891/pandas-plotting-integration-with-matplotlib

###stat prob. space vs biological prob. space
#http://www.gwern.net/Lewis%20meditation
#http://blog.nextgenetics.net/?e=85
#http://www.dspguide.com/ch34/1.htm --benford

# r vs python -- graphs
#http://www.theswarmlab.com/r-vs-python-round-3/ --> ci for regression plots

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
from random import random
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
 
from pylab import *
from numpy import loadtxt
from scipy.optimize import leastsq
from scipy.stats import sem
import csv
from itertools import izip
import os.path
import scipy as sp
import pandas as pd
import itertools
from collections import namedtuple
import time
import matplotlib.animation as animation

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
		self.bins = None	#bayes blocks intervals
 		self.entbys = None
		self.data=[]
		#self.data[0] = dv; self.data[1]= da; self.data[2]=dr
		self.data.append(dv); self.data.append(da); self.data.append(dr);
		self.df = None #panda dataframe
		self.dw = pd.DataFrame(self.wts)
		self.alrmV =None
		self.he=None

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


#tele_raw_dta = np.recfromcsv("/home/solver/data/raw-labeled-th/all.csv", dtype=dd)
#mimic_raw = np.recfromcsv("/home/solver/Desktop/data/raw-labled-mimic/all.csv", dtype=dd)
					

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
	

		"frame"
		hdr = [ i[0] for i in self.data[0] ]
		df2 = pd.DataFrame(self.data[2], columns=hdr[:6] ) #remove Label
		#print 'lbl', len(self.data[2][:,5])
		df2['Label']=self.data[2][:,5]
	

		#set columns
		"incorrect" 
		df2['INC'] = self.inc
		
		"entropy -- return heic"
		df2['ENT'] = self.entbys  #print 'e len\n' , self.entbys[-5:-1],'\n', df2.tail(5)
		ff = df2['ENT'].order()
		print 'ent', len(ff), '\n', ff.tail(10)
		
		
		##hdi
		#"--histogram kde plotkdefit()"
		#mu = df2['ENT'].mean()
		#si = df2['ENT'].std()
		##tl= "r'$\mathrm{Histogram\ of\ Entropy:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, si)"
		##df2['ENT'].hist(bins=self.bins, color='k', alpha=0.3, normed=True)
		##df2['ENT'].plot(kind='kde',style='k--', xlim=[0,(df2['ENT'].max()+5)] , title='entropy over weights')
		#
		#"sort self.entbys, get hdi -> 95ci, all other points h"
		#df2['srt'] = df2['ENT'].sort
		##hdi = lambda x: (x isin 
		##df2['hdi'] = df2['ENT'].map(hdi)

		"easy-way just set a lambda set non-zero to hard"
		he = lambda x: x>0 and 'h' or 'e'
		df2['HE'] = df2['ENT'].map(he)
		print 'he\n', df2['HE'].value_counts()
		self.he = df2['HE'].values

	   #TODO 2013-10-16 11:42
	   # make function that takes hdi and uses that as 95%ci


		#self.dw
		"weights"
		# 4150x50 df['WTS'] = self.wts
		print 'wts shape', self.wts.shape
		print self.dw.head(10)

### hard easy
# ecdf difference in two distributions(hard-easy)

# hdi

# gaussian-mixture: set threshold on entrop; delta reference: map(weight -> entropy).  inverse fit-function get weight from threshold entropy.  what is reference?(healthy, no alert); can visualize separate groups(observed label) based on delta histogram; gaussian-mixture. fit prior(unobserved + label) distribution using expectation-maximization(result is the posterior, iterate)
		
# MCMC sampling: prior(beta distr) look at diff. of thetas(alert/non); likelihood function(bernoulli rv prob of theta1 or theat2) given observed data; combine prior and likelihood to calc the posterior using MCMC (burn, metropolis, etc.) make histogram to see if difference (HDI) between thetas.
		#self.df['alr1SYS'] alr1+ vitals	
		#sem (standard error measure) conf. interval
		#alerts create columns for each header 1>ci 0<ci
		"--alerts" 
		for h in hdr[:6]:
			se = sem(df2[h])
			#(df2[h].std() / df2[h].count() * 1.96 #z from cum dis fun (cdf)
			n = df2[h].count(); #nn = df2[h].shape[0]; nnn=len(df2[h])
			#print n, nn, nnn
			z = se * sp.stats.t._ppf((1.95)/2., n-1)

			#print 'ci', ci
			maxci= df2[h].mean() + z
			minci= df2[h].mean() - z
			#print 'maxmin', maxci, minci
			df2['alr1'+h]= df2[h].map(lambda x: ( x > maxci and '1') or (x < minci and '1') or '0' ) 

		#print df2['SYS'].tail(30)
		print df2[df2['alr1SYS'] == 0 ].count()

		#self.alrm 
		"select row over all alr1-types as series using xs"
		"concat string->alert type"
		hdr = [ i[0] for i in self.data[0] ]

		#print 'hdr ', hdr
		df2['alrm'] = df2['alr1'+hdr[0]]
		for h,i in enumerate(hdr[:6]):
			if h != 0:
				df2['alrm'] = df2['alrm'] + df2['alr1'+i]  #'alrm' equals all rows in one ie 000100

### matching product(n^k) to combinations (n choose k permuationats divided by 2, ie symmetric) by iterating over the 1...k combinations over n {n choose 1}, {n choose 2}...{n choose k}.  using dict of product-space to iterated-sum combination space.  These are held in the dataframe object as 'alrm' and 'alrmV'.  'alrm' column is a tuple, an instance of namedtuples(Vtl) is created over its values.  The Vtl named-tuples are the key, map to the valus of alrmV; a dict{ key named-tuple: value is alarm combo tuple.  So input is tuple -->{dict} --> output is tuple.  
		#self.df.alrmV
		"alert label from 101000 to sysdiaox"
		#alarms = ['sys-','dia-','hr1-','ox-','hr2-','wht-')
		alm = df2['alrm'].values
		codeprd = [i for i in itertools.product('01',repeat=6)] #('1','1')
		#print 'alarm ' , alm[:10] #111111

		#almperm = [ combinations(hdr[:6],repeat=i) for i in range(6) ]
		a1= [ii for j in range(0,7) for ii in itertools.combinations( hdr[:6],j)]
		a0 =['no_alert']
		alarmcmbo = a0 + a1[1:]
		#print 'prod', len(codeprd), codeprd[-5:]
		#print 'combo', len(alarmcmbo), alarmcmbo[-5:] 
	
		"named tuples key(1,1,1,1,1: a,b,c,d)"
		Vtl = namedtuple('Vtl',['sys','dia','hr1','ox','hr2','wht'])
		keytuplperm =[ Vtl(*i) for i in codeprd ]
		dee = dict( izip(keytuplperm, alarmcmbo) )

		#print 'dee ' , dee[tuple('111111')]
		df2['alrmV'] = df2['alrm'].apply(lambda x: dee[tuple(x)] )
		#print 'alrmV ', self.df.alrmV.values[:10]
		self.alrmV = df2['alrmV'].values
		####
		"dataframe class instance attribute"
		self.df = df2
		#print self.df.describe()
##############################################################



# alrmV group over values by sex,geography,timeOfday
# plot weighted bubble graph: color=incorrect, y-axis entropy, x-axis reading-index , plot threshold of hard vs easy: indicator function
	def barz(self):
		"--barplot"
		"frame"
		bz = pd.DataFrame({'he': self.he, 'inc': self.inc, 'alrmV':self.alrmV } )
		print '**alrmV', self.alrmV[:10], bz.alrmV.value_counts()
	
		"parse data"
		#confs/dmhi-current/reports/.txt	
		"group/count unique/ sort -> value_counts"
		#self.df['Label'].idx(1).count()
		
		
		"counts"
		at = pd.value_counts(bz.alrmV); #print "**alert-types-10\n", at.shape, at[:10]
		inc = bz.inc.value_counts(); #print "incorrect\n", inc #1=incorrect
		he = bz.he.value_counts(); #print "hardeasy\n", he
		#inc.plot(kind='bar')

		"group by"  #gender, geography, timeofday
		grouped = bz.groupby(['he','inc'])#.sum().plot(kind='bar', stacked=True)		
		#key = [k for (k,v) in grouped.groups]
		#print 'key', key
		#print grouped.size()
		#print 'PPP', grouped.value_counts()
		pew=grouped['alrmV'].value_counts().unstack().fillna(0.)
		print 'heic vals(\n' 
		pprint.pprint(pew)

		pew.plot(kind='bar',stacked=True)
		#heic = [izip(k,v.sum) for (k,v) in grouped]
			
	def cumsumplt(self):
		#cum-sum
		#cs = self.df['Label','HE']
		#self.df.HE(np.cumsum)

		#plot: sex geog time
		#fig = plt.figure(); 
		#fig,axes = plt.subplots(1,3,sharex=True, sharey=True) #"idiom, index=1 but axes is indexed at zero"
		#plt.subplots_adjust(wspace=0,hspace=0)	
		#axes[0,0].pd.value_counts(self.df.alrm).plot(kind='bar')    
			

		"legend -- self.df['alrmV']"

	

		show()
		#plt.savefig('barplot.svg')
		#plt.savefig('barplot.png', dpi=400, bbox_inches='tight')

	def bblplt(self, entk):
		"--bubbleplot"
		"x-axis: index"	
		fig = plt.figure();
		fig,axes = plt.subplots(2,2)
		#fig,axes = plt.subplot2grid((2,3))

		lin = self.df.index.shape[0]
		self.df['ptrd']= range(lin )
		
		"text: set INC 1 to 'xxx', CORR 0 to 'o'"
		#print 'inc', len(self.df.i2), i2[:10]
		#text(self.df.ptrd.values, self.df.ENT.values, \
#				 self.df['i2'].values, \
#				 size=11, horizontalalignment='center')
		
		"size of weights"	
		sm = self.dw.sum()
		#ct=sm.value_counts()
		#print 'count sum', ct
		print 'dw sum ', sm.values[-10:]	
		sz = [s/.0001 for s in sm.values]
		print 'dw sum/.01 ', sz[-10:]

		"x=readings ,y=entropy "
		x=[]; y=[]
		#self.df['ENT'].values
		data = entk 
		i=0
		for d in data: #done for matplotlib reasons, does not like np, only wants lists
			x.append(i) 
			y.append(d)
			i+=1

		dd = self.df.HE.values
		d2= [1 if d=='e' else 0 for d in dd]
		print 'd2', len(d2), d2[0:10]	
		"incorrect"
		#inc=lambda x: x==1 and 'xxx' or 'o'
		#self.df['i2'] = self.df['INC'].apply(inc)
		ii = self.df.INC.values
		ccc = self.df['INC'].value_counts().values #3747 403
		print 'ccc ', ccc[0], ccc[1]
		iz= [ 0 if i==0 else 1 for i in ii] 
		print 'ii',len(ii), ii[0:10]
		print 'iz',len(iz), iz[0:10] #sm2 = self.dw.astype(float).sum()/ self.dw.astype(float).sum().max()	

		"jitter weights"
		yy=[]; zz =0
		ysum = sum(y)
		for yz in y:
			zz = random() * 5.0
			#yy.append((yz)/ysum)
			yy.append(yz+zz)
			#yy.append(log(yz))
		#print 'yy ' , min(yy),max(yy)
		
		"bbl-plot: x=itr,y=entr,c=heic, s=weight cmap='autumn'"
		#c=d2
		print 'x-range', len(x), len(yy)
		#axes[0,0].scatter(x[0:1000],yy[0:1000], c=iz, s=sz , cmap='autumn',alpha=0.9 ) #linewidths=2, edgecolor='w')
		#axes[0,1].scatter(x[1001:2000],yy[1001:2000], c=iz, s=sz , cmap='autumn',alpha=0.9 )
		#axes[1,0].scatter(x[2001:3000],yy[2001:3000], c=iz, s=sz , cmap='autumn',alpha=0.9)
		#axes[1,1].scatter(x[3001:len(x)],yy[3001:len(x)], c=iz, s=sz , cmap='autumn',alpha=0.9)

		"loop for bubble-plot 4-layout"
		xmin=0
		xmax=1000
		for i in [0,1]:
			for j in [0,1]:
				axes[i,j].scatter(x[xmin:xmax], yy[xmin:xmax], c=iz[xmin:xmax], s=sz[xmin:xmax], cmap='autumn', alpha=0.9)
				xmin+=1001 #ymin+=1001;izmin+=1001;szmin += 1001
				xmax+=1000 #ymax+=1000;izmax+=1000;szmax += 1000

		#"styling"
		plt.subplots_adjust(wspace=.2,hspace=.2)
		
		"axis" #( [0,xmax,ymin,ymax] )
		xlabel('patient readings')
		ylabel('entropy')
		#xlim(0,4150)
		#ylim(0,70)
		
		"title: hard easy"
		hhh = self.df['HE'].value_counts()
		h1 = hhh[1]; hh2 =hhh[0]
		cr = ccc[0]; ccr=ccc[1] 
		title("boosted weights vs entropy CRR=%s,INC=%s,HARD=%s,EASY=%s" % (cr,ccr,h1,hh2))
		#plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))
		#plt.title(r'$\mathrm{boosted\ weight\ vs\ entropy:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))
		
		"legend"
		legend(loc='best')
		axes[0,0].set_label('correct=%s'%(cr))
		axes[0,0].set_label('incorrect=%s'%(ccr))

		"kde histogram"
		"add graph separate title, axis, etc"
		show()



		"y=count x=binned-entropy c=INC s=weight"


		"-- parallel box-plot to compare hard vs easy"
		
		
		"--time-series plot"







		
#	def kdeCI(self): pass
#	def germanTank(self): pass #"think stats book"
#
#	def alert1(self):
#		"confidence intv"
#
#		"get frame, row"
##		self.df[self.data[


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

	def animoBbl(self):
		"--bubbleplot"
		"size of weights"	
		fig = plt.figure();
		plt.axis([0,4150,0,70])

		sm = self.dw.sum()
		sz = [s/.0001 for s in sm.values]
		#ct=sm.value_counts()
		#print 'count sum', ct
	
		x=[];yy=[]; y=[]
		iz=[]
		plt.ion()
		plt.show()
		for anm in xrange(13):
			if anm==0: continue #entropy is 0 messes up
			
			"get entropy"
			data = self.animoEntropyK(anm)
			for i,d in enumerate(data): #done for matplotlib reasons, does not like np, only wants lists
				x.append(i) 
				y.append(d)

			dd = self.df.HE.values
			d2= [1 if d=='e' else 0 for d in dd]
			print 'hard/easy', len(d2), d2[0:10]	
			"incorrect"
			ii = self.df.INC.values
			ccc = self.df['INC'].value_counts().values #3747 403
			print 'corr/Inc ', ccc[0], ccc[1]
			iz= [ 0 if i==0 else 1 for i in ii] 

			"jitter weights"
			zz =0
			ysum = sum(y)
			for yz in y:
				zz = random() * 5.0
				yy.append(yz+zz)
			
			"plot: x=itr,y=entr,c=heic, s=weight cmap='autumn'"
			if anm == 1:
 				#scat = plt.scatter(x,yy, c=iz, s=sz , cmap='autumn',alpha=0.9 ) 
				scat=plt.scatter(x,yy, c=iz, s=sz , cmap='autumn',alpha=0.9 ) 
				#plt.show()
				#plt.imshow(scat)
				#plt.draw()	
			print 'setting data', anm
			#plt.pause(0.5)
			ani = animation.FuncAnimation(fig, self.scatF ,frames=13, fargs=(scat, x,yy,iz,sz), blit=True)	
			x=[];yy=[]; y=[]
			print 'draw'
			#time.sleep(0.05)
			#plt.draw()	
			plt.show()	
		"axis" #( [0,xmax,ymin,ymax] )
		xlabel('patient readings')
		ylabel('entropy')
		#xlim(0,4150)
		#ylim(0,70)
		
		"title: hard easy"
		hhh = self.df['HE'].value_counts()
		h1 = hhh[1]; hh2 =hhh[0]
		cr = ccc[0]; ccr=ccc[1] 
		title("boosted weights vs entropy CRR=%s,INC=%s,HARD=%s,EASY=%s" % (cr,ccr,h1,hh2))
		
		"legend"
		legend(loc='best')
		axes[0,0].set_label('correct=%s'%(cr))
		axes[0,0].set_label('incorrect=%s'%(ccr))

		"kde histogram"
		"add graph separate title, axis, etc"

	def animoEntropy(self):
		plt.ion()
		ct = self.wts.shape[0]
		entK= [ [] for i in xrange(ct-1)] #copy is shallow, nested not copied
		print 'entK ', entK[-1], len(entK), ct
		for k in xrange(ct-1):
			if k==0: 
				print 'pass1'; 
				continue;
			entFeatures = self.wts[:k,:]
			print 'entFeatures k -- ' , entFeatures.shape[0]
			c = Counter()
			for e in set(entFeatures.flat):
				c[e]+=1
			entT = map(list, zip(*entFeatures))
			ent = [self.entCntBin( e ) for e in entT ]
			#print 'ent k -- ' , len(ent)	
			entK[k].append(ent)
		print 'ent k -- done'
		return entK	

	def animoEntropyK(self,k):
		entFeatures = self.wts[:k,:].copy()
		print 'entFeatures k -- ' , entFeatures.shape[0]
		c = Counter()
		for e in set(entFeatures.flat):
			c[e]+=1
		entT = map(list, zip(*entFeatures))
		ent = [self.entCntBin( e ) for e in entT ]
		#print 'ent k -- ' , len(ent)	
		return ent
	
	def scatF(self,k,scat,x,y,iz,sz):
		scat = plt.scatter(x,y, c=iz, s=sz , cmap='autumn',alpha=0.9 ) 
		plt.draw()
		#scat.set_offsets(y)
		#scat.set_offsets(iz)
		#scat.set_offsets(sz)

    	#scat.set_array(ent)
		#print 'ent k -- ' , len(ent)	
		return scat

	#print 'entK', entK[:3]
		#fig = self.bblplt(entK[0])
		#print 'animo'
		#for j in xrange(ct):
			#if j==0: break
			#time.sleep(0.05)






		

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
		print "HEYY##"
		#kernel = stats.gaussian_kde( ent )
		#kdeX = np.linspace(0,5,100)
		#thresh = [ kernel(x) for x in kdeX ]
		#he = [   i for t in thresh ]
		
		#plot(kdeX,kernel(kdeX),'r',label='kde') # distribution function
		fig = plt.figure();
		fig,axes = plt.subplots(1,1)
		
		mu = np.mean( self.entbys )
		si = np.std( self.entbys )
		tl= "r'$\mathrm{Histogram\ of\ Entropy:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, si)"
		##df2['ENT'].hist(bins=self.bins, color='k', alpha=0.3, normed=True)
		##df2['ENT'].plot(kind='kde',style='k--', xlim=[0,(df2['ENT'].max()+5)] , title='entropy over weights')
		density = kde.gaussian_kde(self.entbys)
		xgrid = np.linspace(min(self.entbys), max(self.entbys), 1000)
		plt.hist(self.entbys, bins=8, normed=True)
		plt.plot(xgrid, density(xgrid), 'r-')
		plt.title=tl
		plt.show()

	def plotkdefit(self):
		# best fit of data
		(mu, sigma) = norm.fit(self.entbys)
		
		# the histogram of the data
		n, bins, patches = plt.hist(self.entbys, bins=self.bins, normed=1, facecolor='green', alpha=0.75)
		
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
# parallel plot (violin plot)
#http://pyinsci.blogspot.com/2009/09/violin-plot-with-matplotlib.html

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



### stats-scores
# aor (tibshirani paper)
# simpsons paradox http://vudlab.com/simpsons/
# kelly criterion

# bayes kruscke model diagrams
#http://doingbayesiandataanalysis.blogspot.com.au/2012/05/graphical-model-diagrams-in-doing.html


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
