'''
# countdown
a. alr fatigue, real-time threshold
b. biosurveillance as prior art
1. cli: %_method,credible_interval, fft, online(sprt), alert pie{kern-reg, 
2. weights: make red the hard
3. bayes: monyhallgame in d3(betty drawing), kreske, event-model:pr(#)+t.t.event


# parse the telehealth html files
~/confs-papers-sldes/DMHI-2012-current/reports/*.html  (51)
~/confs-papers-sldes/DMHI-2012-current/server-files txt json files incomplete (33)... but may just be due to low data count

~/BOOKS/bk2/BOOKS/kingston082011
# load pandas --wide-format
sex, age, date-time, sys,dia,hr1,ox,hr2,wht 

-- melt to long date-time, vitals, val, subject_id
# meta-table
subject_id, meds, sex,
gend = [m,f,m,,....]
sid = [1,2,3,4,]
meta = DataFrame(gender, index=sid)
vtl.join(meta,on=key)
'''

import pylab
import pandas as pd
import random
import csv
import dateutil
import os
import numpy as np
from operator import add, sub

from scipy import stats
import statsmodels.api as sm
import scipy  
import scikits.bootstrap as bootstrap
from math import log
from scipy.stats import lognorm, norm

import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot


#from statsmodels import api as sm

# -- test merge vitals and metadata
vtl = pd.DataFrame({'subject_id': range(10), 'vals':np.random.randn(10)})
gndr = [ random.choice(['M','F']) for i in range(10)]
sid = range(10)
meta = pd.DataFrame(gndr, index=sid, columns=['sex'])
vtl = vtl.join(meta, on='subject_id')

#load 
# -- ~/project/data/load_data_th/reports2/*.txt csv
#.txt files
header = ['fname','lname', 'dt1', 'sys', 'dia', 'hr1', 'dt2', 'ox', 'hr2', 'dt3', 'wht']
drcty = '/home/solver/project/data/load_data_th/reports2/'

# -open files in dir
# -- extract Data: section
# Meds: Data: Notes:
start = False
out = []
for fls in os.listdir( drcty):
	fls = os.path.join(drcty,fls)
	#print'FILES ', fls
	with open(fls ,'r') as f:
		rdr = csv.reader(f, delimiter='\t')
		for row in rdr: 
			if 'data:' in [r.lower() for r in row]: 
				start = f.tell()
				continue
			if 'notes:' in [r.lower() for r in row]: 
				start = False
			if start:
				out.append(row)
# -- csv writer
flw = drcty + 'all.csv'
with open(flw, 'wb') as csvfile:
	pndwrt = csv.writer(csvfile, delimiter=';')
	pndwrt.writerows(out)
	
# -- load dataframe from array
#th_data = pd.DataFrame(out, columns=header)
#th_data.columns = header 
th_data = pd.read_csv(flw, encoding='latin1', names=header, sep=';', parse_dates=True, dayfirst=False, index_col='dt1')
th_data.index.names=['realtime']
#print 'TH ', th_data.head() , type(th_data), th_data.dtypes

# - clean data --(null) na
def cleanr(x):
    if x==r"NULL":
        return "NaN"
    else:
        return x
th_data = th_data.applymap(lambda x: cleanr(x))
#any(th_data is "NULL")
th_data = th_data.dropna()
any( pd.isnull( th_data) )

#---fun,#life,news, to read when done------------------------------------------------------------------------------------------------------------------------
#cancer project compuational glover virus kill cancer, ecosystem
#http://www.washingtonpost.com/news/morning-mix/wp/2014/05/15/womans-cancer-killed-by-measles-virus-in-unprecedented-trial/?tid=hp_mm

#gaming ouya is good $50
#http://wololo.net/2014/05/16/the-shock-of-playing-the-ouya-one-year-later/

#library books
#https://medium.com/book-excerpts/4ca8405f1e11

#google voice howto
#http://blog.gleitzman.com/post/40774573324/liberating-google-voice-placing-and-receiving-calls
#rubics cube
#http://fulmicoton.com/posts/rubix/
# http://www.sciencenews.org/view/generic/id/345820/description/Delaying_gratification_is_about_worldview_as_much_as_willpower","Delaying gratification is about worldview as much as willpower | Psychology | Science News

#http://puzzlers.org

#auto-poem http://con.puzzlers.org/mainecon/flats

# bias in narrative
#http://www.psmag.com/navigation/books-and-culture/game-telephone-way-hear-enemies-blame-80301/

# security
#http://glenngreenwald.net/pdf/NoPlaceToHide-Documents-Uncompressed.pdf
#surv vs espion : https://www.schneier.com/essay-449.html
#http://www.net-security.org/secworld.php?id=16694
#http://eprint.iacr.org/2014/257.pdf
#https://tails.boum.org/
#heartbleed
#http://blog.meldium.com/home/2014/4/10/testing-for-reverse-heartbleed
#-----------------------------------------------------------------------------------------------------------------<



#---private data---------------------------------------------------------------------------
#bootstrap data make private
#http://healthyalgorithms.com/2013/10/08/statistics-in-python-bootstrap-resampling-with-numpy-and-optionally-pandas/

#fatigue overtreatment
#http://www.healthmetricsandevaluation.org/news-events/seminar/overdiagnosed-making-people-sick-pursuit-health


from private import private_data
th_data = private_data(th_data)

# - parse time
#--TODO-----
# -- avg time (dt1,dt2,dt3) 
# take timedelta of max3 - min3
# then divide that by 2, add it to min3, convert back to time
##def avgT(x):
##	#time delta object has days
##	return (x['dt1'] + x['dt2'] + x['dt3'] / 3 )
##th_data['avgT'] = th_data.apply(avgT) #, axis=1)
##
##print('$$ ', th_data['avgT'].dytpe )
# -- parse date time
#print 'date ', th_data['dt1'].head()
#th_data['dt1'] = th_data['dt1'].apply( lambda x: dateutil.parser.parse(x ) )
#pd.to_datetime( th_data['dt1'] )
#print ('&&dt ', th_data['dt1'].dtype, type(th_data['dt1']) )
#-----------

#long format
# -- melt
#th_data = pd.melt(th_data, value_vars=[('sys','dia','hr1','ox','hr2','wht')]) #, var_name='vitals', value_name='vals' )
th_data_stack = th_data.stack()
#print 'TH stacked ', th_data.head(5) #, th_data.describe()

#---industry------------------------------------------------------------------------------------
#http://careers.stackoverflow.com/jobs/55871/senior-principal-software-data-engineer-dc-audax-health?a=168IKtple
#jobs

#leap motion api to track motion ** project idea ?
#using for hand tremor in comments using for neuro company
#http://www.christopherlsmith.com/projectblog/2014/5/13/hand-tremor-analysis

#oscar hacking obamacare 80mln funding
#http://oscarhealth.tumblr.com/

#simplyinsured sanfran ca
#http://www.jobscore.com/jobs2/simplyinsured/software-developer/d9IZfSd10r46BliGakhP3Q?detail=Hacker+News&remail=&rfirst=&rlast=&sid=161

#http://omop.org/TechnologyRequirements

#farm android
#https://farmlogs.com/jobs/

#python security
#http://careers.stackoverflow.com/jobs/54799/android-developer-viaforensics?a=14PF3RDpK

#civis datascience
#https://www.linkedin.com/jobs2/view/13095256?trk=job_view_browse_map

#software-engineering
#books
#https://news.ycombinator.com/item?id=7756497
#howdoi
# howdoi format time in python

#good code to read
#http://docs.python-guide.org/en/latest/writing/reading/
#https://github.com/gleitz/howdoi/blob/master/howdoi/howdoi.py

'''
bad: 
def a(*args):
	x,y=args	
	return dict(**locals)
def b(x,y ):
	return({'x':x,'y':y}) 
'''	

#python broad guide
#http://docs.python-guide.org/en/latest/

# optimization
#http://blog.regehr.org/archives/1146

#poetry challenge June 1
# http://con.puzzlers.org/mainecon/flats
#pointer challenge
#https://blogs.oracle.com/ksplice/entry/the_ksplice_pointer_challenge
#generators
#http://stackoverflow.com/questions/17688435/python-how-to-append-generator-iteration-values-to-a-list
#http://ozkatz.github.com/improving-your-python-productivity.html","Improving Your Python Productivity 
#http://www.drewconway.com/zia/?p=1614
#http://www.drewconway.com/zia/?p=204

#>-------------------------------------------------------------------------------------------------<

# - time interval
def timeplotH(dt, title='mimic'):
	#group subject_id
	# -- reset index
	dt.reset_index(inplace=True)
	dt['realtime'] = dt['realtime'].apply(pd.to_datetime )   
	#print 'type ', dt.dtypes 
	#print type(dt.realtime)

	# --  min, max timestamps
	grpt = dt.groupby('subject_id')['realtime']
	subj_min  = grpt.min() ; 
	subj_max  =  grpt.max() ;
	descr = grpt.describe() #df['delta'] = (df['tvalue']-df['tvalue'].shift()).fillna(0)
	
	
	# subject min max 
	mn = subj_max - subj_min
	intdy = mn.map(lambda x: int(x.days) < 7 and 7 or int(x.days) < 365 and int(x.days) or int(0) )
	
	#mean stats
	mm = mn.map(lambda x: int(x.days) )
	mu = mm.mean()
	ms = mm.std()
	#print('day as int**: ', mm, mu, ms )
	#print('#deltaT', mn) #print('days ', mn.map(type) ) 
	
	#global min max
	# -- y is number of subjects, 
	y = list( xrange( dt.subject_id.nunique() ) )
	# -- x is days int 
	mmin = dt.realtime.min();
	mmax = dt.realtime.max()
	mlen = mmax - mmin;
	totdys = int(mlen.days)
	
	#plot model:: [mmin ..  (min ..  x1/2 .. max) .. mmax]
	subrange = subj_min - mmin
	subrdays = subrange.map(lambda x: int(x.days) )   #print('##! ', len(subrdays), subrdays[:10]) 
	x5raw = subrdays + intdy/2
	x5raw.sort()
	x5 = x5raw.map(lambda x: x%365 )  #int(x) < 365 and int(x) or int(365/2))
	
	# plot dashed year-lines
	# -- use x5raw as no of days -> year 
	x55 = x5raw.copy()
	
	#chop up the x55 days
	x55.sort() 
	xdz = [ d/365 for d in x55]
	yrs = []; xz =0
	for i,x in enumerate(xdz):
		if x != xz:
			yrs.append(i)
			xz=xz+1

		
	#print x55
	#print('###chopped ', xdz )
	#print('###years ', yrs )

	# plot
	sns.axes_style("darkgrid")
	fig, ax = plt.subplots()
	ax.errorbar( x5, y, xerr=intdy, fmt='ok', ecolor='grey',elinewidth=2, alpha=0.5)
	ax.set_xlabel("365_days, day_range");ax.set_ylabel("subjects incr_by_year");
	#for i,x in enumerate(xdz):
	#   ax.plot( [0,i] , [365,i], 'k_', lw=1 )
	#   ax.text(1, -0.6, r'$\sum_{i=0}^\infty x_i$', fontsize=20)
	ax.set_title(r'$%s \/\ days : \/\ \mu = %0.0f ,\/\ \sigma = %0.0f$' % (title, mu , ms), fontsize=25)
	#for i,y in enumerate(yrs):
	#	ax.text(-0.1, y, 'year%d' % i, fontsize=12) # ha='center', va='center')	

	#plt.axis=((0,500,0,45))
	#fig.autofmt_xdate()
	#plt.show()
	


#--plotting------------------------------------------------------------------------------------------------------------------
#perl 
#http://en.wikipedia.org/wiki/Xmgrace
#**VERY-GOOD matplotlib tutorial/summary: GOTO
#http://www.loria.fr/~rougier/teaching/matplotlib/
#rgba vals:
#http://matplotlib.org/api/colors_api.html
#colors
#http://stackoverflow.com/questions/8931268/using-colormaps-to-set-color-of-line-in-matplotlib
#matplotlib plot
#http://matplotlib.org/1.3.1/users/pyplot_tutorial.html

#gallery
#http://leejjoon.github.io/matplotlib_astronomy_gallery/

# very nice radar type plot
## * http://leejjoon.github.io/matplotlib_astronomy_gallery/cfasurvey/cfasurvey.html

#pendulum animation
#http://www.moorepants.info/blog/npendulum.html

#subplots
#http://nbviewer.ipython.org/github/duartexyz/BMC/blob/master/Ensemble%20average.ipynb

#seaborn light dark
#http://www.stanford.edu/~mwaskom/software/seaborn/tutorial/aesthetics.html

#plot of patients as stick figures:
# http://nbviewer.ipython.org/gist/theandygross/4544012

#story-telling
#https://github.com/LLK/scratch-flash

#--------------------------------------------------------------------------------------------------------------------------------<

#---time-series analysis-----------------------------------------------------------------------------------------------------------
#fitting model to long time series
#http://robjhyndman.com/hyndsight/long-time-series/
#time series cv 
#"http://alumni.cs.ucr.edu/~mvlachos/","Michalis (Michail) Vlachos Homepage","ude.rcu.sc.inmula.",2,0,1,,1669,1369873364885084,"1c1k9c_f4KX9"
#"http://alumni.cs.ucr.edu/~mvlachos/taships.html","academics","ude.rcu.sc.inmula.",1,0,0,,117,1369873313788325,"G8UcT0qnkHAN"

#apache logs of website correlate server_request with traffic
#http://nbviewer.ipython.org/github/koldunovn/nk_public_notebooks/blob/master/Apache_log.ipynb

# sun-spots time normalization
#http://nbviewer.ipython.org/gist/jhemann/4569783

#ordering of time series for ML
#http://stats.stackexchange.com/questions/3337/ordering-of-time-series-for-machine-learning?rq=1
#http://robjhyndman.com/hyndsight/crossvalidation/

#time series regression
#http://statweb.stanford.edu/~jtaylo/courses/stats203/notes/time.series.regression.pdf

#make time stationary
#http://stats.stackexchange.com/questions/2077/how-to-make-a-time-series-stationary?rq=1


#---alerts-----------------------------------------------------------------------------------------------------------------------------<

'''
#median-deviation
#http://stackoverflow.com/questions/22354094/pythonic-way-of-detecting-outliers-in-one-dimensional-observation-data
#power-law
https://pypi.python.org/pypi/powerlaw
#fft
http://see.stanford.edu/see/courseInfo.aspx?coll=84d174c2-d74f-493d-92ae-c3f45c0ee091
#bayes log-normal model
http://engineering.richrelevance.com/bayesian-ab-testing-with-a-log-normal-model/
#Alerts
# - kernel_reg, change_pt, fft, counsyl, prediction_interval
#otter2tmp
# raw, multivariate, kernreg, fft

#fitting log-normal
#http://stackoverflow.com/questions/8747761/scipy-lognormal-distribution-parameters
#http://stackoverflow.com/questions/8870982/how-do-i-get-a-lognormal-distribution-in-python-with-mu-and-sigma?lq=1

#confidence interval
#http://www.variousconsequences.com/2010/02/visualizing-confidence-intervals.html
#http://stackoverflow.com/questions/18792752/bootstrap-method-and-confidence-interval?rq=1
# fft
#/refs/music.pdf mexican-hat transform
#http://ozkatz.github.com/improving-your-python-productivity.html

# distribution/pareto or percolation
#http://www.liafa.univ-paris-diderot.fr/~labbe/blogue/2012/12/percolation-and-self-avoiding-walks/

#bayesian changepoint
#https://github.com/hildensia/bayesian_changepoint_detection.git
#http://comments.gmane.org/gmane.comp.python.scikit-learn/10332

#event related data analysis
#http://www.pymvpa.org/tutorial_eventrelated.html

#failure rate
http://healthyalgorithms.com/2014/05/16/mcmc-in-python-estimating-failure-rates-from-observed-data/

#sequential
http://nbviewer.ipython.org/github/rasbt/algorithms_in_ipython_notebooks/blob/master/ipython_nbs/sequential_selection_algorithms.ipynb?create=1

#powerlaw
#http://nbviewer.ipython.org/gist/jeffalstott/19fcdd6a4ba400ce8de2

# vienese maze
bostock d3.js maze to tree

#anomoly detection
http://stats.stackexchange.com/questions/10271/automatic-threshold-determination-for-anomaly-detection?rq=1

#outlier
318009,"http://stats.stackexchange.com/questions/1142/simple-algorithm-for-online-outlier-detection-of-a-generic-time-series?rq=1","Simple algorithm for online outlier detection of a generic time series - Cross Validated","moc.egnahcxekcats.stats.",0,0,0,4730,45,,"PDGpQ6B2hWZ6"
318011,"http://stats.stackexchange.com/questions/5700/finding-the-change-point-in-data-from-a-piecewise-linear-function?rq=1","regression - Finding the change point in data from a piecewise linear function - Cross Validated","moc.egnahcxekcats.stats.",0,0,0,4730,45,,"WPYKWjdDaE-u"
318013,"http://stats.stackexchange.com/questions/35137/appropriate-clustering-techniques-for-temporal-data?lq=1","machine learning - Appropriate clustering techniques for temporal data? - Cross Validated","moc.egnahcxekcats.stats.",0,0,0,4730,45,,"sksniL5X1R6D"
318019,"http://biomet.oxfordjournals.org/content/92/4/787.abstract","Symmetric diagnostics for the analysis of the residuals in regression models","gro.slanruojdrofxo.temoib.",0,0,0,,45,,"LZEwqHIqyLnv"
318027,"http://www.originlab.com/www/helponline/origin/en/UserGuide/Graphic_Residual_Analysis.html#Detecting_outliers_by_transforming_residuals","Graphic Residual Analysis","moc.balnigiro.www.",0,0,0,11632,45,,"ywvhk5nvwpT_"

#3d { singlevscum, kernel_reg, change_pt, fft }
#pie graph

#multivariate copulas
http://nbviewer.ipython.org/github/olafSmits/MonteCarloMethodsInFinance/blob/master/Week%209.ipynb?create=1

#clustering
#condesnsation generalized p(x) track in clutter
http://www.robots.ox.ac.uk/~misard/condensation.html
'''

# Confidence interval is the mean,variance over a population; so that given 100 random samples and a 95%CI, 5 CIs should be expected to not contain the normal mean(0) variance(1).  Prediction interval is the probability the next point is within the population mean,variance. 

#http://statsmodels.sourceforge.net/devel/examples/generated/example_ols.html	
def confidenceinterval(dt):
	'''
	#http://nbviewer.ipython.org/github/duartexyz/BMC/blob/master/ConfidencePredictionIntervals.ipynb
	calculate all samples mean, std
	calculate ci for 1 sample group by subject_id
	
	lambda apply to dt['alertci'] if not contain mean, then alert
	count number of alerts as tuple for each 	

	if deviate plot as red
	#variables
	- inputvectors6 1D-array ['sys', 'dia', 'hr1', 'ox', 'hr2', 'wht']
	- runfunction 
	- output plot
	- join axis
	'''
	iv6 = ['sys', 'dia', 'hr1', 'ox', 'hr2', 'wht']
	iv6count = [ i + 'cicount' for i in iv6 ]
	print iv6, iv6count
	vitalcols = [ dt[c] for c in iv6 ]
	#countcols = [ dt[c] for c in iv6count  ]

	# -- splat-unpack	
	#alertcount = NamedTuple('alertcount', [subject_id, dataindex] + iv6 )
	#index=12, dtp_val = [1,0,0,1,1,1]; subjid = 13456; 
	#alertcount(subjid, index, *dtp_val) 

	out = []
	# iv6, iv6count are just list of name-strings
	for vc,cnt in zip(iv6,iv6count):
		print '?? val cols ', dt[vc].head()
		#population
		n = dt['subject_id'].count()      # total number of observations, all subject_id
		M = dt[vc].mean() 
		S = dt[vc].std()  
		T = stats.t.ppf(.975, n-1)        # T statistic for 95% and n-1 degrees of freedom

		#groupby subject_id
		x = dt.groupby('subject_id')[vc]  # 1 subject_id
		m = x.mean()      
		s = x.std()		  				  # panda default axis=0, ddof=1  
	
		#confidence interval given population parameters
		# -- column name variables
		vcz = vc + 'z'		#z-normalized columns
		vcs = vc + 'cisub'; vca = vc + 'ciadd'; #ci add subtract columns
		vco = vc + 'outlier'

		# -- z-transform mean=0 variance=1
		dt[vcz] = dt[vc].map( lambda x: (x- M ) / S)
		Mz = dt[vcz].mean()
		Sz = dt[vcz].std()

		def ci(arr, op):
			rhs = arr * T / np.sqrt(n)
			return np.array( op(Mz, rhs) ) 

		dt[ vcs ] = dt[[vcz]].apply( ci, axis=1,op=sub )
		dt[ vca ] = dt[[vcz]].apply( ci, axis=1,op=add )

		 # CIs that don't contain the true mean
		def cint(x):
			return 1 if x[vcs]*x[vca] > 0 else  0
		#app2 = lambda x,y: x*y >0 and 1 or 0
		dt[vco] = dt.apply(cint,axis=1)

		#print '\n z-norm ', dt[vcz ][:3]
		#print '\n confidence interval ', dt[vcs ][:3]
		#print '\n confidence interval ', dt[vca ][:3]
		#print '\noutliers ', dt[ dt[vco] == 1]

			
		#bootstrap ci
		#http://stats.stackexchange.com/questions/18396/determining-the-confidence-interval-for-a-non-normal-distribution
#http://stackoverflow.com/questions/16707141/python-estimating-regression-parameter-confidence-intervals-with-scikits-boots
#http://stats.stackexchange.com/questions/92209/can-i-pull-a-confidence-interval-out-of-a-single-sample-by-dividing-it-into-sub
		#el = sm.emplike.DescStat(dt[vc])
		#print 'ci ', el.ci_mean()
		# tibshirani resample with replacement at same sample size
		vcob = vco + 'b'
		CI = bootstrap.ci(dt[vcz], scipy.mean, alpha=0.10 ) #, n_samples=10000)
		print '\n bootstrapped 90%', CI[0], CI[1], len(CI), CI
		dt[vc+'ci'] = [ 1 if i>CI[0] and i<CI[1] else 0 for i in dt[vcz] ]
		print 'dtvc ', dt[vc+'ci'][:10], '\n ', dt[vc][:10], len( dt[dt[vc+'ci']==1])/len(dt) * 100.0



	#plot	--dt['whtci']	
	fig, ax = plt.subplots(1, 1, figsize=(13, 5))
	ind = np.arange(1, 101)
	ax.axhline(y=0, xmin=0, xmax=n+1, color=[0, 0, 0])
	ax.plot([ind, ind], CI[:100], color=[0, 0.2, 0.8, 0.8], marker='_', ms=0, linewidth=3)
	ax.plot([ind[out], ind[out]], ci[:, out], color=[1, 0, 0, 0.8], marker='_', ms=0, linewidth=3)
	#ax.plot([ind[out], ind[out]], ci[:, out], color=[1, 0, 0, 0.8], marker='_', ms=0, linewidth=3)
	ax.plot(ind, m, color=[0, .8, .2, .8], marker='.', ms=10, linestyle='')
	ax.set_xlim(0, 101)
	ax.set_ylim(-1.1, 1.1)
	ax.set_title("Confidence interval for the samples' mean estimate of a population ~ $N(0, 1)$",
             fontsize=18)
	ax.set_xlabel('Sample (with %d observations)' %n, fontsize=18)
	plt.show()

	
#confidenceinterval(th_data)

#recursive generators
#http://linuxgazette.net/100/pramode.html
#def boostrs(vc):
#	obsno = len(vc)
#	i=0
#	while T:
#		if i==0:
#			c0=np.random.choice(vc, size=obsno, replace=True )
#			yield c0
#			i = 1;
#			continue
#		c1 = np.random.choice(c0, size=obsno, replace=True )	
#		yield c1
#		c0,c1 = c1, c1.next()
#
#def boostrs2(vc):
#	obsno = len(vc)
#	vc = vc[:] #local copy
#	i=0
#	if i==0:
#		c0=np.random.choice(vc, size=obsno, replace=True )
#		yield c0
#		i = 1;
#	c1 = np.random.choice(c0, size=obsno, replace=True )	
#	yield c1
#	c0,c1 = c1, c1.next()
#
#def boostvc( dt, header, samples=100 ):
#	#1x100 array [mean,std]
#	for vc in header:
#		bvc = 'boost' + vc 
#		bvc = [];
#		for s in range(samples):
#			rs = boostrs2(dt[vc])
#			rsmean = rs.mean() # np.mean(rs, axis=0)
#			rsstd = rs.std() # np.std(rs, axis=0)
#			bvc.append([rsmean, rsstd])
#
#iv6 = ['sys', 'dia', 'hr1', 'ox', 'hr2', 'wht']
#boostvc(th_data, header=iv6)

#bootstrap procedure vs t-test
#*** very good ! http://courses.washington.edu/matlab1/Bootstrap_examples.html
#http://www.stat.umn.edu/geyer/5601/examp/
#prediction interval vs confidence interval:
#**S.Thrun stat lecture on CI
''' ci is population(within) pi is p(next) <t+1> in interval '''

def boostci(dt, samples=100):
	'''generates 100 bootstrap samples from observations'''
	'''returns frame with columns=[mean,std,vitals,cip,cis]'''
	''' ci is based on percent cutoffs, so ['ci'] just shows range '''
	''' to normalize use T-stat  #T = stats.t.ppf(.975, n-1)a '''
	iv6 = ['sys', 'dia', 'hr1', 'ox', 'hr2', 'wht']
	dtlen = len(dt)	

	for v,vc in enumerate(iv6):
		### dt-frame columns ### 
		bci = 'boostci'+vc #mean std
		bco = 'boostoutlier' + vc #1 if not in mean	
		
		#obs boostraped over 1000 samples 
		meantemp=[];stdtemp=[] 
		for i in xrange(samples):
			if i == 0:
 	   			c0 = np.random.choice(dt[vc], size=dtlen, replace=True)
				c1 = c0
	 	   		meantemp.append( np.mean(c0,dtype=np.float64) )
				stdtemp.append( np.std(c0, dtype=np.float64) ) 
 	   			continue;

 	   		c0,c1 = c1, np.random.choice(c0, size=dtlen, replace=True)
 	   		meantemp.append( np.mean(c1,dtype=np.float64) )
			stdtemp.append(  np.std(c1, dtype=np.float64) )

		#ci
		ciptemp = np.add( meantemp,np.sqrt(stdtemp)) 		
		cistemp = np.subtract( meantemp,np.sqrt(stdtemp)) 		
	
		#ci(95%) drawn from empirical_distribution(samples=1000) using \
		#t-distr, to norm the samples'(e.d) std(+-),samples'(e.d) mean 
		n = samples 
		T = stats.t.ppf(.975, n-1)
		print '**T ', T
		Mz=meantemp
		#def cif(arr, op=sub):
		#	rhs = arr * T / np.sqrt(n)
		#	return np.array( op(Mz, rhs) ) 
		#def cif2(mn, op):
		#m,s = meantemp,stdtemp
		#ci = m + np.array([-s*T/np.sqrt(n), s*T/np.sqrt(n)])
		#out = ci[0, :]*ci[1, :] > 0       # CIs that don't contain the true mean
		#print 'out ', out
		#tstatsub 
		
			
		#tstatsub = apply(cif,meantemp,axis=1,op=sub )
		#tstatadd = apply(cif,meantemp,axis=1,op=add )
		#tstatplus = meantemp.apply( cif, axis=1,op=add )
				

		#DataFrame object: columns=mean, std, vital: create/append
		#'tip':tstatplus, 'tis':tstatsub,
		if v==0:
			vc = [vc for i in xrange(samples) ] #gen 100 vcs downcolumn
			dtboost =pd.DataFrame({'mean':meantemp,'std': stdtemp, 
									'vitals':vc,  								 							   'cip':ciptemp, 'cis':cistemp, 
								})
		elif v>0:
			print 'VVV ', v
			vc = [vc for i in xrange(samples) ] #gen 100 vcs downcolumn
			new =pd.DataFrame({'mean':meantemp,'std': stdtemp, 
									'vitals':vc,  								 							   'cip':ciptemp, 'cis':cistemp, 
								})
			dtboost = pd.concat([dtboost,new], ignore_index=True)

	print 'dtboost frame ', dtboost.head(), dtboost.tail(), dtboost.describe()
	return dtboost

def logboostci(dt, samples=100):
	''' generates 100 bootstrap samples from observations
		** 1.log(x) : 2.B[mean +- T*std] 3.vs normal_mean
		1.x : 2.b[log(mean) +- log(T)*std] 3.vs mean
	#log-normal is not log of normal, but distribution whose log is normal
	#log-mean no mu, exp(mu+std^2/2)
	http://nbviewer.ipython.org/url/xweb.geos.ed.ac.uk/~jsteven5/blog/lognormal_distributions.ipynb
	'''
	dtlen= len(dt)	
	vc ='wht'
	whtv = dt[vc].values

	#z-transform
	mm = np.mean(whtv,axis=0)
	ss = np.std(whtv,axis=0)
	def z(x):
		return (x-mm)/(2.0 * ss)
	whtvz=[z(x) for x in whtv]
	#print 'log z ', whtv, whtvz, mm,ss

	#log(x)
	logwht = [ np.log(x) for x in whtv ]
	#print '**log t ', dt.head(),'\n', vc2.head()
	#print 'log wht ', logwht
	
	#bootstrap 
	meantemp=[];
	stdtemp=[] 
	for i in xrange(samples):
		if i == 0:
 			c0 = np.random.choice(logwht, size=dtlen, replace=True)
			c1 = c0
	   		meantemp.append( np.mean(c0,dtype=np.float64) )
			stdtemp.append( np.std(c0, dtype=np.float64) ) 
 			continue;

 		c0,c1 = c1, np.random.choice(c0, size=dtlen, replace=True)
 		meantemp.append( np.mean(c1,dtype=np.float64) )
		stdtemp.append(  np.std(c1, dtype=np.float64) )

	
	#ci(95%) 
	n = samples 
	t1 = stats.t.ppf(.975, n-1)
	T = stats.t.ppf(.975, n-1,loc=mm,scale=ss)
	#print '**T ',t1, T, np.log(T)
	#T = np.log(T)
	T = t1

	m=np.array(meantemp) / log(mm) 
	s=np.array(stdtemp)	/ log(ss) 
	#print 'm,s ** ','\n', m[:3], '\n', s[:3], 
	#m,s=meantemp,stdtemp
	#ci = m + np.array([-s*T/np.sqrt(n), s*T/np.sqrt(n)])
	cis = m - s*T/np.sqrt(n)
	cip = m + s*T/np.sqrt(n)
	ci = np.array([cis,cip])
	#print '\nci\n',ci
	out = ci[0, :]*ci[1, :] > 0       # CIs that don't contain the true mean
	ot = cip-cis
	#out = np.power(cip-cis,2) > 0
	print 'out ', ot[:3]
	
	#plot
	fig, ax = plt.subplots(1, 1, figsize=(13, 5))
	ind = np.arange(1, 101)
	ax.axhline(y=0, xmin=0, xmax=n+1, color=[0, 0, 0])
	ax.plot([ind, ind], ci, color='grey', marker='_', ms=0, linewidth=2)
	ax.plot([ind[out], ind[out]], ci[:, out], color=[1, 0, 0, 0.8], marker='_', ms=0, linewidth=3)
	ax.plot(ind, m, color='grey', marker='.', ms=10, linestyle='')
	#ax.set_xlim(0, 101)
	#ax.set_ylim(-1.1, 1.1)
	ax.set_title("Confidence interval: t-test, log(x), bootstrap$",
	             fontsize=18)
	ax.set_xlabel('Sample (with %d bootstrap)' %n, fontsize=18)
	plt.show()
	
#scipy
#http://pages.physics.cornell.edu/~myers/teaching/ComputationalMethods/python/arrays.html
def test(dt):
	''' input is mean(logx) for each subject split by gender'''
	''' get mu,sigma from lognorm fit of data'''
	whtv = dt
	print 'whtv ', len(whtv), whtv.shape

	#GEOMETRIC MEAN exp() anti-log
	MN = np.exp( np.mean(whtv )) #geometric mean,std
	STD = np.std( whtv )
	print 'globals ', MN, STD

	#fit frozen distribution
	dist=lognorm([STD],loc=MN)

	#samples #100x27
	#log( lognorm-x ) fits a normal distribution
	n = 100 ; ls=27  
	#m =  np.log( np.mean( whtv,axis=1 ) )
	#s =  np.log( np.std( whtv, axis=1, ddof=1) )
	m = np.mean( whtv, axis=1 )
	s = np.std( whtv, axis=1, ddof=1)
	print 'std ', s[:3]
#shape
	T = stats.t.ppf(.975, n-1,loc=MN,scale=STD)        # T statistic for 95% and n-1 degrees of freedom
	t2 = stats.t.ppf(.975, n-1 )        # T statistic for 95% and n-1 degrees of freedom
	print 'T ', T, t2
	#T=t2
	#T = np.exp(T)
	ci = np.exp( m + np.array([-s*T/np.sqrt(n), s*T/np.sqrt(n)]) )
	#ci = np.array([m/(s**2), m*(s**2)])
	out = ci[0, :]*ci[1, :] > 0       # CIs that don't contain the true meano
	fig, ax = plt.subplots(1, 1, figsize=(13, 5))
	ind = np.arange(1, n+1)
	ax.plot(ind, dist.pdf(ind),color=[0,0,0] )
	ax.axhline(y=0, xmin=0, xmax=n+1, color=[0, 0, 0])
	ax.plot([ind, ind], ci, color=[0, 0.2, 0.8, 0.8], marker='_', ms=0, linewidth=3)
	ax.plot([ind[out], ind[out]], ci[:, out], color=[1, 0, 0, 0.8], marker='_', ms=0, linewidth=3)
	ax.plot(ind, m, color=[0, .8, .2, .8], marker='.', ms=10, linestyle='')
	ax.set_xlim(0, n+1)
	#ax.set_ylim(-1.1, 1.1)
	ax.set_title("Confidence interval for log ~$logN$",
			             fontsize=18)
	ax.set_xlabel('Sample (with %d observations)' %n, fontsize=18)
	plt.show()

## percent_method
#http://www.uvm.edu/~dhowell/StatPages/Resampling/BootstCorr/bootstrapping_correlations.html
#if time-normalization possible, invariant to it Effron

#a data-object with only 1 vital(grouped) is passed
def percent_method2( dtb1 ):
	'''sorts means, takes top/bottom 5% as cutoff, compute CI-range'''
	dtb = dtb1[['mean','cip','cis','std','vitals']].copy()

	#normalize to [-1 to 1] #sort ascending
	#zscore = lambda x: (x-x.mean() ) / (2.0 * x.std())	
	#dtb = dtb[['mean','cip','cis','std']].apply(zscore)
	dtb = dtb.sort(['mean']) #ascending 

	#raw-sizes
	size = int( len(dtb) )
	print 'size ', size
	praw = int( len(dtb)*0.03 )  
	qraw = len(dtb) - praw

	#index
	index = dtb[['mean']].index 
	dms = dtb[['mean','cip','cis','std']]
	bottom = index[:praw]
	top = index[qraw:]
	mdl = index[praw:qraw]

	#row-selects top bottom middle 
	tix = dms.ix[ top ]
	bix = dms.ix[ bottom ]
	mrand = np.random.choice( mdl, size=len(mdl), replace=False ) 
	mix = dms.ix[ mrand ]

	#append 1 for color outliers percent method
	pl = pd.concat([bix,mix,tix], axis=0) 
	dms['clr'] = dms.index.map(lambda x: x in mdl and 'NaN' or 1)
	print 'dms ', dms.head(), dms.describe()

	
	#normalize to 100 samples
	norm = len(dtb) / 100
	p = int( (len(dtb)*0.03) / norm ) ; q = 100 - p  #25
	sz = 100 - (2*p)
	b= np.random.choice( bottom, size=p, replace=False)
	m= np.random.choice( mdl, size=sz, replace=False)
	t= np.random.choice( top, size=p, replace=False)
	bi,mi,ti = dms.ix[b], dms.ix[m], dms.ix[t]
	ndt = pd.concat([bi,mi,ti],axis=0)

	#shuffle
	print 'p q ', p, q
	print ndt.head(), ndt.tail()
	print ndt.describe()
	return ndt

def ttestboost(dtb1):
	'''
	#t-stat means test(may not be same as %pm) over the empirical dist.
	#std of empirical distribution
	#looking at sample means(ms), and adding the %T(p-val from population)
	#* the std (which is the subset_sample_ mean)
	#test at mean=0 cross 
	sample mean estimate of population mean
	'''
	dtb = dtb1[['mean','cip','cis','std','vitals']].copy()
	#grab 100
	m100 = np.random.choice(dtb['mean'].values ,size=100, replace=False)
	s100 = np.random.choice(dtb['std'].values ,size=100, replace=False)

	#z-normalize the e.d. means and stds
	mm = [m100.mean(),s100.mean()]
	ss = [m100.std(),s100.std()]
	def zscore(x, vl=0):
		return (log(x)-mm[vl] ) / (2*ss[vl])	
	def zscore2(x, vl=1):
		return (log(x)-mm[vl] ) / (2*ss[vl])	
	#m100 = dtb[['mean']].apply(zscore)
	#s100 = dtb[['std']].apply(zscore)
	m100 = [ zscore(i,vl=0) for i in m100]
	s100 = [zscore2(i,vl=1) for i in s100]
	print 'lens %%% ', len(dtb), len(m100), len(s100)	
	print 'weird ', m100[:5], s100[:5]
	
	#raw-sizes
	n = 100
	#m = dtb['mean'].values #the means of e.d
	#s = dtb['std'].values #the std of e.d.
	m = m100
	s = s100
	print '%%%555 ', m[:3], s[:3]
	T = stats.t.ppf(.975, n-1)

	#ci at T% for p-counts at n-d.o.f
	#ci = m + np.array([-s*T/np.sqrt(n), s*T/np.sqrt(n)])
	#out = ci[0, :]*ci[1, :] > 0   # CIs that don't contain the true mean
	cip = m + s*T/np.sqrt(n)
	cis = m - s*T/np.sqrt(n)
	out = cip*cis > 0   # CIs that don't contain the true mean
	ci = np.array([ cis,cip ])
	#ci = zip( cis, cip )
	#print 'out!! ', m100[:5], s100[:5], out[:10], len(out)
	print 'ci ** ', cip[:5], out[:5], ci[:3]
	fig, ax = plt.subplots(1, 1, figsize=(13, 5))
	ind = np.arange(1, 101)
	ax.axhline(y=0, xmin=0, xmax=n+1, color=[0, 0, 0])
	ax.plot([ind, ind], ci, color='grey', marker='_', ms=0, linewidth=2)
	ax.plot([ind[out], ind[out]], ci[:, out], color=[1, 0, 0, 0.8], marker='_', ms=0, linewidth=2)
	ax.plot(ind, m, color='grey', marker='.', ms=10, linestyle='')
	#ax.set_xlim(0, 101)
	#ax.set_ylim(-1.1, 1.1)
	ax.set_title("Confidence interval for the samples' mean estimate of a population ~ $N(0, 1)$",
	             fontsize=18)
	ax.set_xlabel('Sample (with %d observations)' %n, fontsize=18)
	plt.show()


def boots( th_data, dtb ):
	''' simple bootstrap over sys '''
	#bootstrap
	print 'dtv 78', th_data.head()
	CIs = bootstrap.ci(data=th_data['sys'], statfunction=scipy.mean)
	print 'bootstrapped CIs ', CIs
	#grab 100
	cis100 = np.random.choice(dtb['cis'].values ,size=100, replace=False)
	cip100 = np.random.choice(dtb['cip'].values ,size=100, replace=False)

	print 'dtv ', dtb.head()
	#plot
	out = dtb['mean'].map(lambda x: (x>CIs[1] and 1 or x<CIs[0]) and 1 )
	
	n=len(th_data['sys'])
	ci = np.array([ cis100, cip100 ])
	print 'ci ', ci[:5]
	m = dtb['mean']
	print 'out ', dtb['mean'].head(5), out
	fig, ax = plt.subplots(1, 1, figsize=(13, 5))
	ind = np.arange(1, 101 )
	#ax.axhline(y=0, xmin=0, xmax=n+1, color=[0, 0, 0])
	# reg bars
	ax.plot([ind, ind], ci, color='grey', marker='_', ms=0, linewidth=2)
	# outlier
	#ax.plot([ind[out], ind[out]], ci[:, out], color=[1, 0, 0, 0.8], marker='_', ms=0, linewidth=2)
	# means
	ax.plot(ind, m, color='grey', marker='.', ms=10, linestyle='')
	#ax.set_xlim(0, 101)
	#ax.set_ylim(-1.1, 1.1)
	ax.set_title("Confidence interval for the samples' mean estimate of a population ~ $N(0, 1)$",
	             fontsize=18)
	ax.set_xlabel('Sample (with %d observations)' %n, fontsize=18)
	plt.show()


import matplotlib.colors as cl
def boostpercentplot( dt ): 
	'''plots the %-method of boosted samples'''
	'''frame columns = mean, cip, cis, std, clr'''
	dt = dt.reset_index()
	#shuffle
	print dt.head()
	#get outlier indexes
	ot = dt[dt['clr']==1]
	print 'out ', ot.index
	out = ot.index

	#sample m,ci	
	ci = np.array([ dt['cis'],dt['cip'] ])
	m = dt[['mean']]

	fig, ax = plt.subplots(1, 1, figsize=(13, 5))
	ind = np.arange(1, 101)
	#ax.axhline(y=0, xmin=0, xmax=n+1, color=[0, 0, 0])
	ax.plot([ind, ind], ci, color='grey', marker='_', ms=0, linewidth=2)
	ax.plot([ind[out], ind[out]], ci[:, out], color=[1, 0, 0, 0.8], marker='_', ms=0, linewidth=2)
	ax.plot(ind, m, color='grey', marker='.', ms=10, linestyle='')
	#ax.set_xlim(0, 101)
	#ax.set_ylim(-1.1, 1.1)
	#ax.set_title("Confidence Interval: boost percent method (norm invariant)$",fontsize=18)
	#ax.set_xlabel('Sample (with %d observations)' %n, fontsize=18)
	plt.show()
	#
	#pass



#	#normalize
#	M = dt['mean'].mean()
#	S = dt['std'].std() 
#	cmin = dt['cis'].min()
#	cmax = dt['cis'].max()
#	rng = cmax - cmin
#	print ' $$ ', rng
#	#rmin = dt['cis'
#	s=dt['std'].std()
#	# to normalize find the mean, and then get difference from each x
#	# divide by range of delta-x's
#	def normrange(x):
#		M = x['mean'].mean()
#		delta = x['mean'] - M
#		dmin = delta.min() ; dmax = delta.max()
#		drange = dmax-dmin / 2	
#		dnrm = delta / drange
	#print 'ss %% ', ss
	#dt = dt[['mean','cis','cip']].apply(lambda x:x-M/rng)
	#dt['mean'] = dt[['mean']].apply(lambda x: x-M )
	#dt['cis'] = dt[['cis']].apply(lambda x: x-S/S )
	#dt['cip'] = dt[['cip']].apply(lambda x: x-S )
	
	#nrm = lambda x: (x-M)/S
	#dt = dt[['mean','cis','cip']].apply( nrm )
	#dt['cis','cip'] = dt[['cis','cip']].apply( nrm )
	#dt['mean'] = dt[['mean']].apply( nrm )
	#dt['cis'] = dt[['cis']].apply(lambda x: x-M/S)
	#dt['cip'] = dt[['cip']].apply(lambda x: x-M/S)
	#def nrmm(dx):
	#	return ( (dx['mean']-M )	/ dx['std'] )
	#dt = dt[['mean','cis','cip']].apply(lambda x: nrmm(x) )

def percent_method( empirical_distribution ):
	'''sorts dtboost.bci and takes top/bottom 5%'''
	# sort e.d.
	edsor = empirical_distribution[:,0].sort()
	# 95%, 2.5% top bottom
	l = len(edsor)
	tindex = l - l*.025  
	bindex = l*.025
	
	# indexes
	btm = [ i for i,ms in enumerate(edsor) if i < bindex ]
	top = [ i for i,ms in enumerate(edsor) if i > tindex ]
	mdl = [ m for m,ms in enumerate(edsor) if m > bindex and m < tindex ]
	
	# select 100 samples with 5 out 95
	top5 = lambda x: random.choice(edsor[ [btm], [top] ])
	md95 = lambda x: random.choice(edsor[[ mdl ]] )
	a100 = [ top5(i) if i<5 else md95(i)   for i in xrange(100) ] 
	#random.shuffle(a100)
	# ~confidence interval as mean+-std
	ci = edsor[0,:] + (-edsor[1,:], edsor[1,:] )
	print 'ci ', ci
	return ci

def toyplotpm(boostci):
	# plot 100 total from empirical_distribution
	n=100; ind = ed[:]
	fig, ax = plt.subplots(1, 1, figsize=(13,5))
	
	ind = np.arange(1, 101)
	
	ax.axhline(y=0, xmin=0, xmax=n+1, color=[0, 0, 0])
	
	ax.plot([ind, ind], ci[:100], color=[0, 0.2, 0.8, 0.8], marker='_', ms=0, linewidth=3)
	
	ax.plot([ind[out], ind[out]], ci[:, out], color=[1, 0, 0, 0.8], marker='_', ms=0, linewidth=3)
		
	ax.plot(ind, m, color=[0, .8, .2, .8], marker='.', ms=10, linestyle='')
	
	ax.set_xlim(0, 101)
	
	ax.set_ylim(-1.1, 1.1)
	
	ax.set_title("Bootstrap Confidence Interval percentile method", fontsize=18)
	
	ax.set_xlabel('100 Samples' %n, fontsize=18)
	
	plt.show()



#-----#distributions---------------------------------------------
#http://pages.stern.nyu.edu/~adamodar/New_Home_Page/StatFile/statdistns.htm#_ftnref2

#stat reference:
#http://inperc.com/wiki/index.php?title=Elementary_Statistics_by_Bluman

#qq-plot rule out normal
#ks to see if normal, lognormal, weibull, etc
#pass gender grouped
#http://stackoverflow.com/questions/15630647/fitting-lognormal-distribution-using-scipy-vs-matlab?rq=1
#http://stackoverflow.com/questions/8870982/how-do-i-get-a-lognormal-distribution-in-python-with-mu-and-sigma?lq=1
#exploratory:
#http://nbviewer.ipython.org/github/herrfz/dataanalysis/blob/master/week3/exploratory_graphs.ipynb

def lognormal(x,sex='male'):
	''' fits data to log-distribution, with fixed parameters '''
	#1.explicit(take log(x) and get m,s); 2.fit a normal distribution to log(x); 3.fit x to lognormal distribution
	#ddof=1 uncensored, uses unbiased estimator(expected err=0)
	x = x['wht'].values
	def lognfit(x, ddof=0):
		x = np.asarray(x)
		logx = np.log(x)
		mu = logx.mean()
		sig = logx.std(ddof=ddof)
		return mu, sig
	#lognorm
	mean, stddev = lognfit(x)
	print '** std, men **', stddev, mean
	dist = lognorm([stddev],loc=mean)
	#fit
	shape,loc, scale = lognorm.fit(x, floc=0)
	d2 = lognorm([scale], loc=loc)
	#plot
	fig, ax = plt.subplots(1, 1, figsize=(6, 5))
	xi=np.linspace(0,10,5000)

	#fit
	plt.plot(xi,d2.pdf(xi), label='log norm fitted')
	
	#original
	ln = len(x)+1
	#n,bins,patches = 
	plt.hist( np.log(x) , ln, normed=1, label='sample log(x)' ) 
	#np.log(x).hist(bins=ln, ax=ax, alpha=.5,normed=1,label='sample,log(x)' )
	plt.plot(xi,dist.pdf(xi), label='lognorm pdf')
	plt.plot(xi,dist.cdf(xi), label='lognorm cdf')

	#title
	ax.set_xlim(4,8 )
	ax.set_title(r"$lognormal \/\ distribution \/\ weight \/\ %s$" % sex,fontsize=18)
	plt.legend(loc='upper right')
	ax.set_ylabel('frequency log(x)')	
	ax.set_xlabel(r'log(x) n= %d' % ln, fontsize=12)
#	ax.set_title(r'$%s \/\ days : \/\ \mu = %0.0f ,\/\ \sigma = %0.0f$' % (title, mu , ms), fontsize=25)
	plt.show()



def logfitcdf( dt ):
	'''
		plots distribution, and fits
	   	plots cdf and fit
	'''
	#sns.axes_style("darkgrid")
	sns.set_palette("deep", desat=.6)
	#sns.set_context(rc={"figure.figsize": (200, 3)})
	sns.set()
	#current_palette = sns.color_palette("coolwarm", 7)
	#sns.set_style("whitegrid")
	#sns.set_context("notebook")
	#sns.color_palette("muted", 8)
	#sns.set_style("dark")
	#sns.despine(left=True)

	#data
	dtraw = dt.copy().values.flatten()
	dtlog = dt['wht'].apply(lambda x:np.log(x) ).values.flatten()

	# Set up the plots
	f, (ax1, ax2, ax3) = plt.subplots(1,3) 
	c1, c2, c3 = sns.color_palette("dark", 4)[:3]	

	#linear
	maxd = dtraw.max()
	bins = np.linspace(0,maxd, maxd+1)
	ax1.hist(dtraw, bins/4, normed=True, alpha=0.5, histtype="stepfilled")

	shape, loc, scale = stats.lognorm.fit(dtraw, floc=0) 
	mu = np.log(scale) 	# Mean of log(X)
	sigma = shape 		# Standard deviation of log(X)
	M = np.exp(mu) 		# Geometric mean == median
	s = np.exp(sigma) 	# Geometric standard deviation

	x = np.linspace( dtraw.min(), dtraw.max(), dtraw.max()+1 )
	ax1.plot(x, stats.lognorm.pdf(x, shape, loc=0, scale=scale), linewidth=3) 

	ax1.set_xlabel('weight(lbs)')
	ax1.set_ylabel('P(x)')
	ax1.set_title('Linear')
	#leg=ax1.legend()	

	#log
	maxl = dtlog.max()
	bins = np.linspace(0,maxl, maxl+1)
	ax2.hist(dtlog, bins*3, normed=True,alpha=0.3, histtype="stepfilled")

	shape2, loc2, scale2 = stats.lognorm.fit(dtlog, floc=0) 
	mu = np.log(scale)  # Mean of log(X)
	sigma = shape 		# Standard deviation of log(X)
	M = np.exp(mu) 		# Geometric mean == median
	s = np.exp(sigma) 	# Geometric standard deviation

	x = np.linspace(dtlog.min(), dtlog.max(), num=400)
	ax2.plot(x, stats.lognorm.pdf(x, shape2, loc=0, scale=scale2), linewidth=3 ) 
	#sns.kdeplot(dtlog,shade=True,ax=ax2)
	ax2.set_xlabel('log weight(lbs)')
	ax2.set_ylabel('P(x)')
	ax2.set_title('Logs')

	#cdf
	ecdf = sm.distributions.ECDF(dtlog)
	x = np.linspace(min(dtlog ), max(dtlog ))
	y = ecdf(x)
	ax3.step(x, y)
	shape, loc, scale = stats.lognorm.fit(dtlog, floc=0) 
	ax3.plot(x,stats.lognorm.cdf(x, shape, loc=0, scale=scale), linewidth=3)
	
	ax3.set_xlabel('log weight(lbs)')
	ax3.set_ylabel('$\sum P(x)$')
	ax3.set_title('Cumulative')
	f.tight_layout()
	plt.show()


def logfitcdfold( dt ):
	'''
		plots distribution, and fits
	   	plots cdf and fit
	'''
	#data
	#sns.axes_style("darkgrid")
	sns.set()
	current_palette = sns.color_palette("coolwarm", 7)
	
	sns.set_style("whitegrid")
	sns.set_context("notebook")
	#sns.color_palette("muted", 8)
	#sns.set_style("dark")
	#sns.despine(left=True)
	dtraw = dt.copy().values.flatten()
	dtlog = dt['wht'].apply(lambda x:np.log(x) ).values.flatten()
	
	#plt.figure( figsize=(12,4.5))
	#raw: hist + fit
	ax1 = plt.subplot(131)
	n, bins, patches = plt.hist(dtraw, bins=100, normed=True)
	#dtraw.hist(bins=100, alpha=.5,label='all male',normed=1) #,cumulative=True)

	shape, loc, scale = stats.lognorm.fit(dtraw, floc=0) # Fit a curve to the variates
	mu = np.log(scale) # Mean of log(X)
	sigma = shape # Standard deviation of log(X)
	M = np.exp(mu) # Geometric mean == median
	s = np.exp(sigma) # Geometric standard deviation

	x = np.linspace(dtraw.min(), dtraw.max(), num=400)
	plt.plot(x, stats.lognorm.pdf(x, shape, loc=0, scale=scale), linewidth=2 )
	ax = plt.gca() # Get axis handle for text positioning
	txt = plt.text(0.9, 0.9, 'M = %.2f\ns = %.2f' % (M, s), horizontalalignment='right', 
	                size='large', verticalalignment='top', transform=ax.transAxes)	
	#plt.xlim(0,150)
	plt.xlabel('weight(lbs)')
	plt.ylabel('P(x)')
	plt.title('Linear')
	leg=ax1.legend()	

	#log
	ax1 = plt.subplot(132)
	n, bins, patches = plt.hist(dtlog, bins=100, normed=True)
	#dtlog.hist(bins=100, alpha=.5,label='all male',normed=1) #,cumulative=True)

	shape, loc, scale = stats.lognorm.fit(dtlog, floc=0) # Fit a curve to the variates
	mu = np.log(scale) # Mean of log(X)
	sigma = shape # Standard deviation of log(X)
	M = np.exp(mu) # Geometric mean == median
	s = np.exp(sigma) # Geometric standard deviation

	x = np.linspace(dtlog.min(), dtlog.max(), num=400)
	plt.plot(x, stats.lognorm.pdf(x, shape, loc=0, scale=scale), 'b', linewidth=2 ) 
	ax = plt.gca() # Get axis handle for text positioning
	txt = plt.text(0.9, 0.9, 'M = %.2f\ns = %.2f' % (M, s), horizontalalignment='right', 
	                size='large', verticalalignment='top', transform=ax.transAxes)	
	#plt.xlim(0,150)
	plt.xlabel('log weight(lbs)')
	plt.ylabel('P(x)')
	plt.title('Logs')
	leg=ax1.legend()	

	#cdf
	ax2 = plt.subplot(133)
	ecdf = sm.distributions.ECDF(dtlog)
	x = np.linspace(min(dtlog ), max(dtlog ))
	y = ecdf(x)
	ax2.step(x, y)
	shape, loc, scale = stats.lognorm.fit(dtlog, floc=0) 
	ax2.plot(x,stats.lognorm.cdf(x, shape, loc=0, scale=scale), '--', linewidth=2)
	
	plt.xlabel('log weight(lbs)')
	plt.ylabel('$\sum P(x)$')
	plt.title('Cumulative')

	plt.show()

	#x = np.linspace(lognorm.ppf(0.01, s), lognorm.ppf(0.99, s), 100)
	#plt.plot(x,cum,'r--')
	#plt.plot(x, lognorm.pdf(x, s), 'r-', lw=5, alpha=0.6, label='lognorm pdf')
	#rv = lognorm(s)
	#plt.plot(x, rv.cdf(x), 'k-', lw=2, label='frozen pdf')


def normalfit(x):
	#normV
	x = x['wht'].values
	nm, ns = norm.fit(x)
	print '** std, men norm **', nm, ns, x[:3]
	#normdist = norm([ns],loc=nm)

	#plot
	ln = len(x)+1
	fig, ax = plt.subplots(1, 1, figsize=(6, 5))
	xi=np.linspace(0,200,500)
	plt.hist( x , normed=1, alpha=0.3 ) 
	#x.hist(bins=ln, ax=ax, alpha=.5,normed=1,label='sample,x' )
	plt.plot(xi,norm.pdf(xi,loc=nm,scale=ns), label='norm fitted')
	#plt.plot(xi,norm.pdf(xi), label='norm original')
	
	ax.set_title("fit normal distribution",fontsize=18)
	plt.legend(loc='upper left')
	plt.show()

#http://stats.stackexchange.com/questions/77752/how-to-check-if-my-data-fits-log-normal-distribution
def qqlog(x):
	'''log-norm take log(x)'''
	x = x['wht'].values
	x = np.log(x)
	print 'x ** ', x
	qqplot( x , dist=stats.t,line='45', fit=True);
	pylab.show()
def qqnorm(x):
	'''gauss.norm , ks-stat'''
	x = x['wht'].values
	print 'x ** ', x
	qqplot( x , line='45', fit=True);
	pylab.show()
	

#def vitalsdistplot(dt):
#	''' male female log-norm fit of vitals '''
#	#g = sns.FacetGrid(tips, row="sex",col="itemid" hue="itemid")
#	#g.map(
#	pass
		
			



def radarplot():
	#http://www.loria.fr/~rougier/teaching/matplotlib/#d-plots
	axes([0,0,1,1])
	N = 20
	theta = np.arange(0.0, 2*np.pi, 2*np.pi/N)
	radii = 10*np.random.rand(N)
	width = np.pi/4*np.random.rand(N)
	bars = bar(theta, radii, width=width, bottom=0.0)
	
	for r,bar in zip(radii, bars):
	    bar.set_facecolor( cm.jet(r/10.))
	    bar.set_alpha(0.5)

	show()	


def predictioninterval(dt):
	'''
	ensembleaverage http://nbviewer.ipython.org/github/duartexyz/BMC/blob/master/Ensemble%20average.ipynb
	time normalize to percent cycle, interpolate
	lambda apply to dt['ea']
	plot all samples
	plot normalized
	plot ensemble

	can use duartexyz or seaborn
	''' 

def ensembleaverage(dt):
	'''
	time normalized
	http://nbviewer.ipython.org/github/duartexyz/BMC/blob/master/Ensemble%20average.ipynb
	'''

import ellipsoid     
def multivariate_prediction_interval(dt):
	''' 
	--bivariate(ellipse) --trivariate(ellipsoid)
	#http://nbviewer.ipython.org/github/duartexyz/BMC/blob/master/PredictionEllipseEllipsoid.ipynb
	#http://stats.stackexchange.com/questions/29860/confidence-interval-of-multivariate-gaussian-distribution
	'''

#multivariate_prediction_interval(th_data)	
		


#dt['cisingle'] = th_data[['sys', 'dia', 'hr1', 'ox', 'hr2', 'wht']].apply(lambda x: single(x))
#print 'SINGLE ', cisingle

def fft(x):
    def detect_outlier_position_by_fft(signal, threshold_freq=.1, frequency_amplitude=.01):
        fft_of_signal = np.fft.fft(signal)
        outlier = np.max(signal) if abs(np.max(signal)) > abs(np.min(signal)) else np.min(signal)
        if np.any(np.abs(fft_of_signal[threshold_freq:]) > frequency_amplitude):
            index_of_outlier = np.where(signal == outlier)
            return index_of_outlier[0]  
        else:                  
            return None
  
  
    outlier_positions = []
    for ii in range(10, y_with_outlier.size, 5):
        outlier_position = detect_outlier_position_by_fft(y_with_outlier[ii-5:ii+5])
        if outlier_position is not None:    
            outlier_positions.append(ii + outlier_position[0] - 5)
    outlier_positions = list(set(outlier_positions))
  
    plt.figure(figsize=(12, 6));    
    plt.scatter(range(y_with_outlier.size), y_with_outlier, c=COLOR_PALETTE[0], label='Original Signal');
    plt.scatter(outlier_positions, y_with_outlier[np.asanyarray(outlier_positions)], c=COLOR_PALETTE[-1], label='Outliers');
    plt.legend();


def cp(x):
	pass

#kernelregression http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gp_regression.html#example-gaussian-process-plot-gp-regression-py
def kernreg(x):
	pass

def time3d(data):
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	#ax = Axes3D(fig)
	
	xs = np.arange( data.shape[0] )
	ax.bar(xs, data.dia, zs=0, zdir='y', color='y', alpha=0.5)
	ax.bar(xs, data.sys, zs=10, zdir='y', color='r', alpha=0.5)
	ax.bar(xs, data.hr1, zs=20, zdir='y', color='g', alpha=0.5)
	#ax.bar(xs, data.ox, zs=30, zdir='y', color='g', alpha=0.5)
	ax.bar(xs, data.hr2, zs=30, zdir='y', color='b', alpha=0.5)
	ax.bar(xs, data.wht, zs=40, zdir='y', color='m', alpha=0.5)


	#ax.set(zlim=(0,300))
	ax.set_xlabel('X')
	ax.set_ylabel(['dia','sys','hr1','hr2','wht'])
	ax.set_zlabel('Z')
	plt.show()

#----boosting-------------------------------------------------------------------------------------
#http://www.metacademy.org/graphs/concepts/boosting_as_optimization#focus=boosting_as_optimization&mode=explore
#http://130.203.133.150/viewdoc/summary;jsessionid=86C1563E69B2FB2794C36B6BE961F95F?doi=10.1.1.170.2812 Fast boosting using adversarial bandits
#http://stats.stackexchange.com/questions/ask","Adaboost feature weight calculation - Cross Validated - Stack Exchange
#http://stats.stackexchange.com/questions/40568/adaboost-feature-weight-calculation - Adaboost feature weight calculation - Cross Validated
#http://stats.stackexchange.com/questions/25699/feature-selection-without-target-variable?rq=1","Feature selection without target variable - Cross Validated
#http://www.pnas.org/content/105/39/14790.full","Higher criticism thresholding: Optimal feature selection when useful features are rare and weak
#https://github.com/jbeard4/pySCION/commit/0b09590e8e6561f13e72a874d2ce60c4ed304fb2
#http://stats.stackexchange.com/questions/23382/best-bandit-algorithmmachine learning - Best bandit algorithm? - Cross Validated

#dataming
#http://stats.stackexchange.com/questions/1856/application-of-machine-learning-techniques-in-small-sample-clinical-studies?rq=1
#http://en.wikipedia.org/wiki/Thompson_sampling","Thompson sampling - Wikipedia,
#http://stats.stackexchange.com/questions/10271/automatic-threshold-determination-for-anomaly-detection?rq=1

#collinearity
#http://learnitdaily.com/six-ways-to-address-collinearity-in-regression-mVodels/

#-------------------------------------------------------------------------------------------------------------



#----model reliability----------------------------------------------------------------------------
#OLS R-language statsmodel
#http://statsmodels.sourceforge.net/devel/examples/notebooks/generated/robust_models_1.html

# http://en.wikipedia.org/wiki/Micromort
#datamining hyperloglog (very good)
#incoming streaming data probabilistic counting by bits not order stats
#http://research.neustar.biz/2012/10/25/sketch-of-the-day-hyperloglog-cornerstone-of-a-big-data-infrastructure/
'''
st like all the other DV sketches, HyperLogLog looks for interesting things in the hashed values of your incoming data.  However, unlike other DV sketches HLL is based on bit pattern observables as opposed to KMV (and others) which are based on order statistics of a stream.  As Flajolet himself states:
'''


#ML application p >> n
#refs/papers/applyingMLtoclnical*.pdf victoria stodden stanford 2008
#http://sparselab.stanford.edu/

#http://nbviewer.ipython.org/github/mwaskom/Psych216/blob/master/week6_tutorial.ipynb
#http://nbviewer.ipython.org/github/unpingco/Python-for-Signal-Processing/blob/master/Compressive_Sampling.ipynb
#http://nbviewer.ipython.org/url/perrin.dynevor.org/exploring_r_formula_evaluated.ipynb
#survival curves
#http://nbviewer.ipython.org/github/CamDavidsonPilon/lifelines/blob/master/docs/Survival%20Analysis%20intro.ipynb
#psych model weibull distribution**
#http://nbviewer.ipython.org/github/arokem/teach_optimization/blob/master/optimization.ipynb
#image models neuro
#http://nbviewer.ipython.org/github/jonasnick/ReceptiveFields/blob/master/receptiveFields.ipynb

#stat basic stats in pandas
#http://www.randalolson.com/2012/08/06/statistical-analysis-made-easy-in-python/
#http://people.duke.edu/~ccc14/pcfb/analysis.html

#overfitting
#http://www.visiondummy.com/2014/04/curse-dimensionality-affect-classification/

#R-formula
#refs/imm6614.pdf statsmodel patsy R-formula slide30_34

#---disease model--------------------------------------------------------------------------------------------------
#clustering
#http://nbviewer.ipython.org/github/herrfz/dataanalysis/blob/master/week4/clustering_example.ipynb
#http://www.windml.org/examples/visualization/clustering.html

#---bayesian--------------------------------------------------------------------------------------------------------
#http://nbviewer.ipython.org/github/twiecki/pymc3_talk/blob/master/bayesian_pymc3.ipynb
# radon example
#http://nbviewer.ipython.org/github/fonnesbeck/multilevel_modeling/blob/master/multilevel_modeling.ipynb
'''
Sound It Out:(math-phonics)
ditta
The Probability of A * Probability of new data | A =   P(new data) * P(A|new data)    


'''


#normal distribution
#sns.set_style("whitegrid")
#data = 1 + np.random.randn(20, 6)
#sns.boxplot(data);
# test statistic: generalized likelihood ratio


#--------------------------------------------------------------
def main():
	#data
	#gendergroup = th_data.groupby(['gender','subject_id'])['wht']

	gender = th_data[th_data['gender']==' m']	##SELECT GENDER
	males = gender[['wht','subject_id']];	
	maleid19 = males[males['subject_id']==19]

	#sns.distplot( males['wht'].values, kde=False, fit=stats.lognorm );
	#sns.distplot( np.log( males['wht'].values), kde=False, fit=stats.norm );

	#genderex = [ 1,1,1,1,1,0,0,0,0,0 ]
	#hexample = pd.DataFrame({'subject_id': range(10), 'vals':np.random.randn(10), 'gender':genderex})
	#g = sns.FacetGrid(hexample, col="gender")	
	#g.map( sns.distplot, "vals", kde=True, fit=stats.lognorm )
	#print 'gender ', hexample.describe()

	header = ['sys', 'dia', 'hr1', 'ox', 'hr2', 'wht']
	vitalmap={'sys':1, 'dia':2 }	
	th_data['gender'] = th_data['gender'].apply(lambda x:x==' m' and str('Male') or str('Female') )
	th_data.reset_index(drop=True,inplace=True)
	thd = th_data.drop(['dt2','dt3'],axis=1 ) #,inplace=True)

	print 'th_data\n ', thd.head() 

	#th_melt = th_data.drop('realtime',axis=0)
	thmelt = pd.melt(thd, id_vars=['subject_id','gender'] ) #, var_name='vitals' )
	print ' dropped ** ', thmelt.head()
	thmelt['value'].map(lambda x : np.log(x) )

	#print 'melt\n ' , thmelt.head()
	#print 'gender type\n ', type(thmelt['value'])

	#pal = dict(Male=
	g = sns.FacetGrid(thmelt, row="gender", hue="gender",col="variable", margin_titles=True, xlim=(0,300), ylim=(0,0.05), despine=True )
	g.map( sns.distplot, "value" , kde=False, fit=stats.norm, 
			kde_kws={"color": "seagreen", "lw": 3, "label": "KDE"}, 
			hist_kws={"histtype": "stepfilled","alpha":0.7},
			fit_kws={"color":".2", "lw": 3}
			); 
	#stats.pearsonr
	#g.map(sns.kde
	#g.set_axis_labels("Total bill (US Dollars)", "Tip");
	#g.set(xticks=[10, 30, 50], yticks=[2, 6, 10]);
	#g.fig.subplots_adjust(wspace=.02, hspace=.02);


	#malesgrp = males.groupby('subject_id')['whtlog'].values
	#print 'data ', maleid19['wht'].values
	#sns.kdeplot(maleid19['wht'].values)
#	log
#	#100x27
#	smp = [  [w[ np.random.randint(1,len(w)) ] for (w) in malesgrp ] for i in range(100) ] 
#	print 'smple ', len(smp)
#	smple = np.asarray( smp)
#	print 'type *', type(smple)#, smple.dtypes



#--------------------------------------------------------------
	if(0):
		#fit lognormal distribution
		m1 = maleid19.copy()
		lognormal(m1)
		normalfit(m1)	
		qqlog(m1)
		qqnorm(m1)

		#average days of sample
		timeplotH(th_data, title='telehealth')
#--------------------------------------------------------------
	if(1):
		m2 = maleid19.copy()
		logfitcdf(males)
		exit(0)
		#logboostci(m2)
		#test(malesgrp)
		test(smple)
		exit(0)
		#confidence interval
		dtb = boostci(th_data) 
		dtv = dtb[dtb['vitals']=='sys']
		print 'dtv check', dtv.tail(), dtv.describe()

		#boots(th_data, dtv)
		pltframe = percent_method2(dtv)
		ttestboost(dtv)
		#boostpercentplot( pltframe )

		#ci = percent_method(empirical_distribution)
		#toyplotpm(ci)
#--------------------------------------------------------------
	

if __name__=="__main__":
	main()




