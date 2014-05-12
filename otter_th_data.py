'''
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

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import csv
import dateutil
import os
import seaborn as sns

# -- test merge vitals and metadata
vtl = pd.DataFrame({'subject_id': range(10), 'vals':np.random.randn(10)})
gndr = [ random.choice(['M','F']) for i in range(10)]
sid = range(10)
meta = pd.DataFrame(gndr, index=sid, columns=['sex'])
vtl = vtl.join(meta, on='subject_id')

#load 
# -- ~/project/data/load_data_th/reports2/*.txt csv
#Charles     Davis   2008-12-26 12:00:00.000 135 72  75  NULL    NULL    NULL    2008-12-26 12:00:00.000 169.10
#Charles     Davis   2008-12-25 22:56:00.000 158 77  70  2008-12-25 22:57:00.000 97  68  2008-12-25 22:56:00.000 173.60
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


#---private data---------------------------------------------------------------------------
from private import private_data
th_data = private_data(th_data)

# - parse time
#TODO
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


#long format
# -- melt
#th_data = pd.melt(th_data, value_vars=[('sys','dia','hr1','ox','hr2','wht')]) #, var_name='vitals', value_name='vals' )
th_data_stack = th_data.stack()
print 'TH stacked ', th_data.head(5) #, th_data.describe()

#plot
# - time interval
def timeplotH(dt, title='mimic'):
	#group subject_id
	# -- reset index
	dt.reset_index(inplace=True)
	dt['realtime'] = dt['realtime'].apply(pd.to_datetime )   
	print 'type ', dt.dtypes 
	print type(dt.realtime)

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

		
	print x55
	print('###chopped ', xdz )
	print('###years ', yrs )

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
	


timeplotH(th_data, title='telehealth')

#--plotting------------------------------------------------------------------------------------------------------------------
#pendulum animation
#http://www.moorepants.info/blog/npendulum.html

#---time-series analysis-----------------------------------------------------------------------------------------------------------
#http://nbviewer.ipython.org/gist/jhemann/4569783
#plot of patients as stick figures:
# http://nbviewer.ipython.org/gist/theandygross/4544012
#apache logs of website correlate server_request with traffic
#http://nbviewer.ipython.org/github/koldunovn/nk_public_notebooks/blob/master/Apache_log.ipynb



#---alerts------------------------------------------------------------------------------------------------------------------------------
'''
#Alerts
# - kernel_reg, change_pt, fft, counsyl, prediction_interval
#prediction interval
http://nbviewer.ipython.org/github/duartexyz/BMC/blob/master/PredictionEllipseEllipsoid.ipynb

#otter2tmp

# percolation
#http://www.liafa.univ-paris-diderot.fr/~labbe/blogue/2012/12/percolation-and-self-avoiding-walks/
# vienese maze
#3d { singlevscum, kernel_reg, change_pt, fft }
#pie graph
# raw, multivariate, kernreg, fft
#changepoint
#https://github.com/hildensia/bayesian_changepoint_detection.git
#http://comments.gmane.org/gmane.comp.python.scikit-learn/10332
#event related data analysis
#http://www.pymvpa.org/tutorial_eventrelated.html

#sequential
http://nbviewer.ipython.org/github/rasbt/algorithms_in_ipython_notebooks/blob/master/ipython_nbs/sequential_selection_algorithms.ipynb?create=1

#powerlaw
#http://nbviewer.ipython.org/gist/jeffalstott/19fcdd6a4ba400ce8de2
'''

def single(x):
	''' pandas z-transform 	'''
	if x > (x.mean() + dt.std()) or dt.x< dt.mean() - dt.std():
		return 1
	else:
		return -1

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

#-model reliability------------------------------------------------------------------------------------------------
#http://nbviewer.ipython.org/github/mwaskom/Psych216/blob/master/week6_tutorial.ipynb
#http://nbviewer.ipython.org/github/unpingco/Python-for-Signal-Processing/blob/master/Compressive_Sampling.ipynb
#http://nbviewer.ipython.org/url/perrin.dynevor.org/exploring_r_formula_evaluated.ipynb
#survival curves
#http://nbviewer.ipython.org/github/CamDavidsonPilon/lifelines/blob/master/docs/Survival%20Analysis%20intro.ipynb
#psych model weibull distribution**
#http://nbviewer.ipython.org/github/arokem/teach_optimization/blob/master/optimization.ipynb
#image models neuro
#http://nbviewer.ipython.org/github/jonasnick/ReceptiveFields/blob/master/receptiveFields.ipynb

#---disease model--------------------------------------------------------------------------------------------------
#clustering
#http://nbviewer.ipython.org/github/herrfz/dataanalysis/blob/master/week4/clustering_example.ipynb

#---bayesian--------------------------------------------------------------------------------------------------------
#http://nbviewer.ipython.org/github/twiecki/pymc3_talk/blob/master/bayesian_pymc3.ipynb
# radon example
#http://nbviewer.ipython.org/github/fonnesbeck/multilevel_modeling/blob/master/multilevel_modeling.ipynb



#normal distribution
#sns.set_style("whitegrid")
#data = 1 + np.random.randn(20, 6)
#sns.boxplot(data);
# test statistic: generalized likelihood ratio



# -- test
print(th_data.head() )
print(th_data.describe() )




