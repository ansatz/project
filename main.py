#wordle
#picloud amazon cluster
"""

<script type="text/javascript" src="jquery-latest.min.js"></script>
<link href="knowlstyle.css" rel="stylesheet" type="text/css" />
<script type="text/javascript" src="knowl.js"></script>

"""


### main
#if __name__ == "__main__"():
	
#from boosting import * 

from score import *
from pandadata import pndsa
import os.path
from scipy import stats, optimize
import seaborn as sns
sns.set(palette='Set2') 

### boost
"""

already run in ~/project/scikit-learn-master/ as
python -m sklearn.ensemble.boost.py
have to set dd , col-headers in data file


"""

# load 
"data  --workaround for module path issues"
w = '/home/solver/project/scikit-learn-master/weight.pkl'
e = '/home/solver/project/scikit-learn-master/entropybasic.pkl'
i = '/home/solver/project/scikit-learn-master/incorrect.pkl'
d = '/home/solver/project/scikit-learn-master/datahdrlbl.pkl'
ab = score(w,e,i,d) #score.py object
pd = pndsa(w,e,i,d) #pandadata.py object inherits from score-class



# bin
"bayesian_blocks is expensive so check"
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
# <a knowl="ref.html">kwl</a>
"fixed-width 1/4150"
ent4150  = ab.entnrm 
"adaptive-width"
entbayes = ab.entropy() 
print 'entropy', sorted(entbayes)[-15:-1]



# hdi density estimation 
"plot"
###ab.plotkde()
#ab.plotkdefit()
#ab.plotkdeNL()


# score.py pandas
#print 'data to csv' , ab.pcklcsv() #write to csv
"set dataframe columns heic"
ab.dframe()
"plot --stackedbar --bubble --timeseries"
###ab.barz()
###ab.bblplt()
print "animo call-- "
ab.animoBbl()

#ab.animoEntropy()

#ab.plotkdefit()


# indicator function
#print 'entbys', ab.entbys, ab.entbys.ndim
#ab.pnd()


# correct-incorrect-bar
#colours = ["#348ABD", "#A60628"]
#
#hic = [100, 100] #hard_inc_corr
#eic = [200, 300 ] #easy_inc_corr
#prior = [0.20, 0.80]
#posterior = [1./3, 2./3]
#plt.bar([0, .7], prior, alpha=0.70, width=0.25,
#		        color=colours[0], label="hard",
# 				        lw="3", edgecolor=colours[0])
#
#plt.bar([0+0.25, .7+0.25], posterior, alpha=0.7,
# 		        width=0.25, color=colours[1],
# 				        label="easy",
#						        lw="3", edgecolor=colours[1])
#
#plt.xticks([0.20, .95], ["Incorrect", "Correct"])
#plt.title("Hard vs Easy pts for Incorrect/Correct Alert")
#plt.ylabel("Number Alerts")
#plt.legend(loc="upper left");
#		
#plt.show()

