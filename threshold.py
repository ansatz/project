#<script type="text/javascript" src="jquery-latest.min.js"></script>
#<link href="knowlstyle.css" rel="stylesheet" type="text/css" />
#<script type="text/javascript" src="knowl.js"></script>


#<!--
#<IMG SRC="i.jpg" ALT="img" WIDTH=100 HEIGHT=90>
#-->




### ** objective: **	<br>

# 1.return a rnkCIlabel -> correct incorrect for each pt-state	<br>

# 2.return a fpCurve(fp vs kernel-size..feature as bag-of-words) -> and do it for 3 curves	<br>

# window-size of smoothing, 	<br>
#   <<<< sequential-time  >>>>>>	<br>

# kernel-size of kernel	<br>
#   <<<<  form of the data >>>>>>	<br>

# ci-size(static)sequence-size(dynamic)	<br>

from entrofunc import *
from boostwrap import *
#`from hashkernel-master/haskernel import *`
from scipy import stats
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from astroML.plotting import hist


### ** _[what] **	<br>
class threshold(object):
	
	# 2. sampling  -- what is a label? 
	# hashkernel.py,  bayescovariance.py(wishart) diseasemodeling.py
    def indicator(self, entropy):
		return fpCrv  #focused on size-changes effect on fp-rate
	
	# 1. alert label;  -- given labels, what does the data tell us
	# score.py, features.py boosting.py, MAB , heic.py
	# what is probability distribution function?
    def label(self):
		return rnkCI  # bar graph, heic1 vs heic2

	# << meta >>: a. sequential, b. global(static)   (recurrence dynamical system)
	# texture.py, crossvalidate.py(autocorrelate), roc-scores , datasmallvsbig
    def recur(self, mapvct): 
		return ste  #static/sequential, optimization( mle, adaptive-bin, word-type )


### ** [how] **	<br>
#DATA <> queue[where,when] .. direct injection objects	<br>

#entropy - timeseries - whats significant	<br> 
# what is the distribution priors
# parametric based appraoch
class bayesInf( threshold ):
	
	def indicator(self, ent ,size, aMiner ): #multi-arm-bandit single vs global
		#size is kde window size	<br>
		#TODO named-tuple	<br>
		return fpCurve  

	def label( self, H_I, aMiner):    #conjugate_analytic vs MCMC, fixed-entropyvsBayesBlockBin
		return rankedCI #correct_incorrect

	def recurr( self, states, aMiner): #static(wishart) vs sequential(ie the hot-hand)   
		return texturesequenceGraph

#what can be learned - form it takes? ; 
class hashKern( threshold ):
	
	def indicator(self, ent, size, aMiner): #probabalistic scraping mixed-int programming
		#size is bit-size
		return fpCurve
	
	def label( self, H_I, aMiner): #kde=estimate prob.density.function, non-parametrically, bandwidth can be determined through cross-validation
									#different regression effect on classifier, hashkernel #regression and correlation
		return rankedCI
	
	def recurr(self, states, aMiner): #plot the np-hard problem of MIP
		return texturedstateGraph

#what is a label? size of CI... show the;  
class ensemble( threshold ):
	
	def indicator(self, sizeCI, aMiner): #game-based kelly criterion
		kernel = stats.gaussian_kde(ent)
		return fpCurve
	
	def label(self, H_I, aMiner):  #stump vs tree vs mab
		#H_I is the entorpy indicator function
		return rankedCI
	
	def recurr(self, state, aMiner):  #static/sequential, optimizationofkelly
		return exploratorytextureGraph

#class language(alert language, transformer, states)

### [WHO] api
class dataMine(object):
   	 def __init__(self, alt_how_cls ):
     		   self.how = alt_how_cls

   	 def fpCurve(self, ent, size ):
     		   return self.how.indicator(ent, size, self)

   	 def rankedCI(self, H_I):
     		   return self.how.label(H_I, self)

   	 def recurrence(self,state):
     		   return texture()

if __name__ == "__main__":
