# -*- coding: utf-8 -*-
from __future__ import division 
#ipython notebook pylab=inline
import tables
import pprint
from itertools import izip,product
from collections import defaultdict, namedtuple
import datetime as dtt
import traceback
import pylab
import math
import pandas as pd
import random
import csv
import dateutil
import os
import numpy as np
from operator import add, sub
from StringIO import StringIO

from scipy import stats
import statsmodels.api as sm
import scipy  
import scikits.bootstrap as bootstrap
import scipy.stats as ss
from math import log
from scipy.stats import lognorm, norm
import statsmodels.formula.api as smf
from datetime import datetime
from sklearn.metrics import r2_score
import offline_changepoint_detection as offcd
from functools import partial

import prettytable    
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.graphics.gofplots import qqplot
# -------------------------------------------------------------------------------------- #
# Section 1 - data
header = ['fname','lname', 'dt1', 'sys', 'dia', 'hr1', 'dt2', 'ox', 'hr2', 'dt3', 'wht']
drcty = '/home/solver/project/data/load_data_th/reports2/'

flw = drcty + 'all.csv'
if flw !=1:
	# extract Data: section
	# Meds: Data: Notes:
	start = False
	out = []
	for fls in os.listdir( drcty):
		fls = os.path.join(drcty,fls)
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
	# csv writer
	with open(flw, 'wb') as csvfile:
		pndwrt = csv.writer(csvfile, delimiter=';')
		pndwrt.writerows(out)
	
# load dataframe from csv
th_data = pd.read_csv(flw, encoding='latin1', names=header, sep=';', parse_dates=True, dayfirst=False, index_col='dt1')
th_data.index.names=['realtime']

# clean data --(null) na
def cleanr(x):
    if x==r"NULL":
        return "NaN"
    else:
        return x
th_data = th_data.applymap(lambda x: cleanr(x))
#any(th_data is "null")
th_data = th_data.dropna()
if any( pd.isnull( th_data) ):
	print 'telehealth not null'

#private (m/f)
from private import private_data
th_data = private_data(th_data)
th_data_stack = th_data.stack()
#clean
vitalmap={'sys':1, 'dia':2 }	

# 2. _mimic2v26
# flatfiles 1 through 6, can import more records at 
# /home/solver/MIMIC-Importer-2.6/
#otter_data.py takes ( parse.py->all.csv {web sql queries} -> all.csv ) -> dataframe
#from otter_data import m2d
#mc_data = m2d()
from mimic_pandas import mimicpostgresql

# a. mimic_pandas.py queries postgresql -> write to csv
drcty2 = '/home/solver/project/data/'
mimicfile = drcty2 + 'mimic2v26_1_6.csv'

# load directly (+ query also)
#mc_data = mimicpostgresql()
if not os.path.exists( mimicfile ) :
	print '\n:::you are querying the database:::\n'
	#dataframe from postgresql
	dt = mimicpostgresql()
	dt.to_csv(mimicfile, sep='\t', encoding='utf-8')
if os.path.exists( mimicfile ):
	print '\nmimic file exists, not querying\n '

# b. load (pre-query) csv to pandas
hdr =['index','subject_id','sex','dob','dod','hospital_expire_flg', \
	         'itemid', 'description', 'charttime', 'realtime', 'value1num',\
	          'value1uom', 'value2num','value2uom']
# ----  %LOAD 1 - 3 ----- #
##MCD##	
tparse = lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S')
#mc_data = pd.read_csv("./data/mimic2v26_1_6.csv", names=hdr, skiprows=1, sep='\t') #nrows=1000
mc_data = pd.read_csv("./data/mimic2v26_1_6.csv", names=hdr, skiprows=1, sep='\t', nrows=3000)
#mc_data = pd.read_csv("/home/solver/project/data/mimic2v26_1_6.csv", names=hdr, skiprows=1, sep='\t', nrows=3000)
year = dtt.timedelta(days=365)
def shift(x):
	x=list(x)
	x[0] = '1'
	x[1] = '9'
	x=''.join(x)
	return x
	
mc_data['timeshift'] = mc_data['realtime'].map(lambda x: shift(x) )
mc_data['timeshift'] = mc_data['timeshift'].apply(tparse)    #  lambda x: pd.to_datetime(x) )
mc_data = mc_data.reset_index().set_index('timeshift')

# clean data --(null) na
if any( pd.isnull( mc_data) ):
	print '\nmimic null \n'

mc_data.reset_index(inplace=True)

def mc_data_clean_csv(filename='merge_clean.csv'):
    directory = '/home/solver/project/data/'
    mimicfile = directory + filename
    
    if not os.path.exists( mimicfile ) :
        print( 'merged/clean mimic+telehealth dataframe -> csv')
        dt = mimicpostgresql()
        dt.to_csv(mimicfile, sep='\t', encoding='utf-8')
        
    if os.path.exists( mimicfile ):
        print '\nmerged datframe->csv file exists\n '

def pri(msg, dt):
   output = StringIO()
   dt.to_csv(output)
   output.seek(0)
   pt = prettytable.from_csv(output)
   print '\n**',msg,'\n',pt

def main():
# -- format data for fnc -- #
	#format
	# -- gender
	th_data['gender'] = th_data['gender'].apply(lambda x:x==' m' and str('Male') or str('Female') )
	mc_data['gender'] = mc_data['sex'].apply(lambda x:x=='M' and str('Male') or str('Female') )

	# -- source
	th_data['source'] = th_data['subject_id'].map(lambda x: 'telehealth') 
	mc_data['source'] = mc_data['subject_id'].map(lambda x: 'mimic') 


	#melt to long-form
	thm = th_data.reset_index(drop=False) #dont drop realtime
	thd = thm.drop(['dt2','dt3'],axis=1 ) #,inplace=True)
	thmelt = pd.melt(thd, id_vars=['subject_id','gender','source','realtime']) #,var_name=['itemid'] )
	thmelt['value'].map(lambda x : np.log(x) )
	#pri('th long', thmelt.head())

	#mimic(long default) , merge
	mcm = mc_data.reset_index(drop=True)
	mc_data['variable']=mc_data['itemid'].map(lambda x:x)
	mc_data['value']=mc_data['value1num'].map(lambda x:x)
	mcd = mc_data.drop(['itemid','value1num','value2num','index','charttime','sex','hospital_expire_flg','description','value1uom','value2uom'],axis=1 )
	#pri('mimic dropped', mcd.tail() )
	mcd.reset_index(inplace=True)
	#print 'mcd type &&&& ', mcd['realtime'].apply(type) 

	mcd = mcd.dropna()
	#print 'mcd type &&&& ', mcd['realtime'].apply(type) 
	if any( pd.isnull( mcd) ):
		print 'mcd dropped not null'
	#merge
	thmi = pd.merge(thmelt,mcd,on=['subject_id','gender','source','variable','value'], how='outer' )
	pri('merged data set', thmi.head() )
	pri('merged mimic check', thmi[thmi['source']=='mimic'][:5])
	#print '*****len-check****', len(thmi[ thmi['source']=='mimic'] ), len(mcd), len(mcm), len(mc_data)
	return thmi
thmi = main()
# ----  %LOAD 2 - 3 ----- #
# @dataset
sns.set_context('notebook') #paper,notebook,poster
sns.set_style("white")      # whitegrid, white, dark,

#select subset by source
pri('thmi',thmi.head())
tt = thmi[ (thmi['variable']=='sys') & (thmi['source']=='telehealth')][['realtime_x','subject_id','value']].reset_index(drop=True)

# -- display raw data, sequence/overlap 
#(to discuss data pooling issues)
# sequence sort time 
pt = tt.subject_id.unique()[:5]
srt = tt.set_index('realtime_x').sort_index()
#concat
cat=[]
for i in pt:
	pi = srt[ srt['subject_id'] == i]
	pi['list'] = list(range( len(pi) ) )
	cat.append(pi)
p10 = pd.concat(cat, ignore_index=False)
tbl=p10.pivot_table('value',rows='list',cols='subject_id')
tle2 = 'UNINDEXED SYSTOLIC DATA FOR 5 PATIENTS'

# raw time-series data
tf = tt.drop_duplicates(['realtime_x','subject_id','value']) 
tst = tf.set_index(['realtime_x','subject_id']).unstack('subject_id')
tst20 = tst.ix[:,:15]
tle = 'TIMESERIES SYSTOLIC DATA FOR 15 PATIENTS'

## @ notebook
# @ datamining
#quotes
'''
“Essentially, all models are wrong, but some are useful”
George Box
“The only way to find out what will happen when a complex
system is disturbed is to disturb the system, not merely to
observe it passively”
Fred Mosteller and John Tukey, paraphrasing George Box
'''


# -- vitals avg, std boxplots
br = thmi[['source','value','variable']]
#vr = br.groupby(['source']).boxplot(by='variable')  # pandas version

# agg() std() per patient
vdd = thmi[['source','subject_id','variable','value']]
agv = vdd.groupby(['subject_id','variable','source'])['value'].agg([('std','std')]).reset_index(['source','variable'])
## notebook
##sns.set_context("poster")
###f, ax = plt.subplots(2,1, figsize=(10,10))
##sns.factorplot('variable',hue='source',y='value',data=br,kind='box', aspect=2, size=4)
##
### std per patient
##sns.factorplot('variable',hue='source',y='std',data=agv,kind='box', aspect=2, size=4) 

# TODO do not have telehealth ages

# -- demographics   
dmgd = thmi.reset_index(drop=True, inplace=False)
dmg = dmgd.drop(['level_0'],axis=1)
pri('dmg', dmg.head() )

dms =dmg.copy()
ddm = dms[['source','gender']]
print ddm.head()
# get age
def birthdeath(b,d):
	'''>90 set to 90'''
	try:
		bint = int(b[:4] ) #bd.map(lambda x: int(x[:4]) ) 
		dint = int(d[:4] ) #dd.map(lambda x: int(x[:4]) )
		age  = dint - bint
		return age
	except:
		pass
sid = dmg.groupby(['subject_id']).first()
sid['age'] = np.vectorize(birthdeath)( sid['dob'], sid['dod'] ) 
sid = sid.reset_index(drop=False) 
pri('age ',sid.head() )
# set geography randomly, weighted
def weighted_choice(choices):
   total = sum(w for c, w in enumerate(choices) )
   r = random.uniform(0, total)
   upto = 0
   for c, w in enumerate( choices):
      if upto + w > r:
         return c
      upto += w
   assert False, "Shouldn't get here"

def geo(x):
	l = ['urban','sub','rural']
	p = dict(enumerate(l) )
	c = [2,2,1]
	n = weighted_choice(c)
	return p[n]
sid['geo'] = sid.index.map(lambda x: geo(x)) 
pri('geo',sid.geo.head() )
#fill telehealth with an age value
sid['agefill'] = sid['age'].map(lambda x: pd.isnull(x) and np.random.random_integers(40,70) or x) 
# set age groups
agelist = ['<40','40-60','>60']
aged = dict(enumerate(agelist) )
sid['ageg'] = sid.age.map(lambda x: x<40 and '<40' or x>60 and '>60' or '40,60')
sag = sid[['source','ageg' ,'agefill','gender','geo','subject_id']]
pri('sag r', sag.head())

'''
#demographics
sns.set_context("poster")
sns.set_style("darkgrid")
pri('sag',sag.head())

#f, (ax1,ax2) = plt.subplots(1,2)
ax1 = sns.factorplot('geo','agefill', hue='gender', col='source',data=sag,dodge=.2, aspect=.75, size=5)
ax2 = sns.factorplot('geo', hue='gender', col='source', data=sag,aspect=1.25, size=3)
ax1.set(xticks=[],xlabel="")
ax2.set(title="")
sns.despine(left=True, bottom=True)

#data pool
ax1 = sns.factorplot('geo','agefill', hue='gender', col='source', data=sag, aspect=.75)
ax2 = sns.factorplot('geo','agefill', hue='gender', col='source', data=sag, aspect=3, size=.25)
ax1.set(xticks=[],xlabel="") 
sns.despine(left=True, bottom=True)
'''
# INTERARRIVAL TIMES

# -- merge mimic and telehealth time index
mt=thmi[ thmi['source']=='mimic']['timeshift']
tt=thmi[ thmi['source']=='telehealth'][ 'realtime_x' ]
thmi['tidx'] = pd.concat( [mt,tt], axis=0)
thmi['timeindex'] = pd.to_datetime(thmi['tidx'])
print type(thmi['timeindex']), thmi['timeindex'].dtype
print thmi.timeindex.head()

dcatt = thmi[['source','subject_id','variable','timeindex','value']]
d = dcatt.reset_index(drop=True).set_index(['timeindex'],drop=False ).copy()
#d['freq'] = d.index.map(lambda x: x and 1 )
print 'd expand' , d.head()

#avg len of monitoring per pt
def tlen(g):
    g['dlen'] = (g['timeindex'].max() - g['timeindex'].min()).days
    g.drop_duplicates(inplace=True)
    return g
 
mtr = d.groupby(['source','subject_id','variable'],as_index=False, group_keys=False).apply(lambda x: tlen(x))
mtr.drop_duplicates(inplace=True)
mtr.reset_index()
print mtr.head()
thmt = mtr[mtr.source=='telehealth']['dlen']
mmmt = mtr[mtr.source=='mimic']['dlen']
mtrcat = pd.concat([thmt,mmmt],axis=1)
print mtrcat.head()
'''
# avg len of time
#print mtr.dlen[:5]
f,(a1,a2,a3) = plt.subplots(3,1)

#ax1.set(vert=False)
#print cn.freq[:15]
#a2=sns.factorplot("freq",kind="point",data=cn,hue="source",size=3,aspect=1.25)#,ax=a1)

a1.hist(cmm.freq.values,normed=True,histtype="stepfilled",alpha=0.7,label="mimic")
a1.hist(cth.freq.values,normed=True,histtype="stepfilled",alpha=0.7,label="telehealth")
a1.set(title="Frequency of Readings")
a1.set_xticks(range(35))
a1.set_xlabel('Readings per Day')
a1.set_ylabel('Normalized Count')
xl = plt.xlim()
print xl[1]
plt.legend(loc='best')
plt.subplots_adjust(hspace=.7)

sns.boxplot(mtrcat, names=['th','mimic'], ax=a2,vert=False)
a2.set(title="Length of Monitoring")
a2.set_xlabel('days')

sns.boxplot(wci, names=['S','M','T','W','Th','F','Sa'], ax=a3)
a3.set(title="Heart Rate_Within Group Average")
a3.set_ylabel('bpm')

'''

#frequency counts per day
# counts format
fr = d.drop_duplicates(['timeindex','source','subject_id','variable']) 
mr = d.drop_duplicates(['timeindex','source','subject_id','variable']) 
#fr = fr.drop(['source'],1)
fr = fr.set_index(['timeindex','source','subject_id','variable'])
fr3 = fr.copy()
fr3 = fr3.unstack() #unstacks variable(last is default) 
fr3.columns = fr3.columns.droplevel(0) #flattens index
fr3 = fr3.reset_index()
fr3['timeindex'] = pd.to_datetime( fr3['timeindex'] )
pri('fr3',fr3.head() )
	
#counts
cn = fr3.copy()
cn['freq'] = cn.index.map( lambda x: x and 1 ) 
print cn.head()
print 'frequency*** ', cn[['freq']][:10]
#split sources
cnmm = cn[cn.source=='mimic']
cnth = cn[cn.source=='telehealth']
#group by subject_id
cmm = cnmm.set_index('timeindex').groupby(['subject_id']).resample('D',how='count')
cth = cnth.set_index('timeindex').groupby(['subject_id']).resample('D',how='count')
###pri( 'cn',cn.head(10) )
print 'frequency*** sampled', cn[['freq']][:10]
cmm = cmm.unstack()
cmm = cmm.reset_index(drop=True)
cth = cth.unstack()
cth = cth.reset_index(drop=True)
cmm = cmm.stack()
cth = cth.stack()

print cmm.head(15)
print cth.head(15)
print 'frequency*** mm ', cmm.freq.values[:10]
print 'frequency*** th', cth[cth.freq.values>0][:10]



#cn['fmean']=cn['freq'].mean()
print cn.freq
print '**** ', cn.freq.values[:5]
print 'mimic',  cn[cn.source=='mimic'][:5]
#print cn.fmean.head()
print cn[cn['freq'].values>3][:5]
print cn.head()

##
print('resampled', cn[:10] )
print '*** len > 3 '	
print len( cn[cn['freq']>3] ), len(cn)

mr = mr.reset_index(drop=False)
print mr.head()
mrg = pd.merge(cn,mr,how='outer') #left_index=True,right_index=True,join='outer')
print mrg.head()

#weekday trend
w = d.copy()
print w.head()
t = pd.DatetimeIndex(w.timeindex)
w['wday'] = t.weekday
print w['wday'].head()
wci = w[w.variable=='hr1'].groupby(['subject_id','wday','variable'])['value'].mean().unstack('wday')
print wci.head()

#timeseries
#acf plot
#periodogram

# scatter plot
print tbl.head()
print d.head()
p19 = d[(d['subject_id']==19) & (d['variable']=='sys')]
p19 = p19.set_index('timeindex').sort_index()
lag0 = p19.value.values[:-1]
lag1 =p19.value.values[1:]
lg = {'lag0':lag0, 'lag1':lag1}
plag = pd.DataFrame(lg)
print plag.head()
print len(p19.variable), len(lag0), len(lag1)

## poly regression
p19['t'] = np.arange(1, p19.shape[0]+1)
print p19.head()
ptrain = p19[['value','t']]
test_p = ptrain.iloc[-10:, :]
train_p = ptrain.iloc[:-10, :]
print train_p.shape, test_p.shape
print ptrain.head()

model = smf.ols(formula="value ~ t + I(t**2)", data = train_p)
model = model.fit()
#print model.summary()
print smf.SummarizeResults(model)
print r2_score(train_X.Ridership, model.predict(train_X ))
print r2_score(test_X.Ridership, model.predict(test_X ))
print 'hello'

# ----  %LOAD 3 - 3 ----- #
#durbin-watson statistic

'''
f,a1= plt.subplots(1,1, figsize=(7,5))
a1 = sns.regplot("lag0", "lag1", plag, fit_reg=False, ax=a1);
a1.set(title="Lag1 AutoCovariance")
a1.set_ylabel(r'$systolic,x_{t1}$')
a1.set_xlabel(r'$systolic,x_{t0}$')
plt.legend(loc='best')
plt.subplots_adjust(hspace=.7)

f,(a2,a3,a4) = plt.subplots(3,1, figsize=(9,5))
a3 = ptrain.value.plot(ax=a3)
#a2.plot_date(ptrain.index, model.predict(ptrain), 'r-')
a3.plot(ptrain.index, model.predict(ptrain), 'r-')

a4.set(title="Residuals")
x=ptrain.value - model.predict(ptrain)
plt.plot(x)
#plt.legend()

a2 = sm.graphics.tsa.plot_acf(ptrain.value - model.predict(ptrain), lags = 10, ax=a2 )

plt.legend(loc='best')
plt.subplots_adjust(hspace=.75)



data.ipynb
#demographics
sns.set_context("poster") #, font_scale=1.25)
sns.set_style("darkgrid")

#f, (ax1,ax2) = plt.subplots(1,2)
ax1 = sns.factorplot('geo','agefill', hue='gender', col='source',data=sag,dodge=.2, aspect=.75, size=5)
ax2 = sns.factorplot('geo', hue='gender', col='source', data=sag,aspect=1.25, size=3)
ax1.set(xticks=[],xlabel="") #,title="Demographics of 2 Patient Groups")
ax2.set(title="")
sns.despine(left=True, bottom=True)
#plt.subplots_adjust(hspace=0)
plt.subplots_adjust(hspace=.5)

# avg len of time
#print mtr.dlen[:5]
f,(a1,a2,a3) = plt.subplots(3,1)

#ax1.set(vert=False)
#print cn.freq[:15]
#a2=sns.factorplot("freq",kind="point",data=cn,hue="source",size=3,aspect=1.25)#,ax=a1)

a1.hist(cmm.freq.values,normed=True,histtype="stepfilled",alpha=0.7,label="mimic")
a1.hist(cth.freq.values,normed=True,histtype="stepfilled",alpha=0.7,label="telehealth")
a1.set(title="Frequency of Readings")
a1.set_xticks(range(35))
a1.set_xlabel('Number per Day')
a1.set_ylabel('Normalized Count')
xl = plt.xlim()
print xl[1]
plt.legend(loc='best')
plt.subplots_adjust(hspace=.7)

sns.boxplot(mtrcat, names=['th','mimic'], ax=a2,vert=False)
a2.set(title="Length of Monitoring")
a2.set_xlabel('days')

sns.boxplot(wci, names=['S','M','T','W','Th','F','Sa'], ax=a3)
a3.set(title="Heart Rate_Within Group Average")
a3.set_ylabel('bpm')

# inter-arrival times
# weibull time to first change
#Decompose time domain
f,a1= plt.subplots(1,1, figsize=(7,5))
a1 = sns.regplot("lag0", "lag1", plag, fit_reg=False, ax=a1);
a1.set(title="Lag1 AutoCovariance")
a1.set_ylabel(r'$systolic,x_{t1}$')
a1.set_xlabel(r'$systolic,x_{t0}$')
plt.legend(loc='best')
plt.subplots_adjust(hspace=.7)

f,(a2,a3,a4) = plt.subplots(3,1, figsize=(9,5))
a3 = ptrain.value.plot(ax=a3)
#a2.plot_date(ptrain.index, model.predict(ptrain), 'r-')
a3.plot(ptrain.index, model.predict(ptrain), 'r-')

a4.set(title="Residuals")
x=ptrain.value - model.predict(ptrain)
plt.plot(x)
#plt.legend()

a2 = sm.graphics.tsa.plot_acf(ptrain.value - model.predict(ptrain), lags = 10, ax=a2 )

plt.legend(loc='best')
plt.subplots_adjust(hspace=.75)


# values
sns.set_context("poster")
#f, ax = plt.subplots(2,1, figsize=(10,10))
sns.factorplot('variable',hue='source',y='value',data=br,kind='box', aspect=2, size=4)

# std per patient
sns.factorplot('variable',hue='source',y='std',data=agv,kind='box', aspect=2, size=4) 

#Data Pooling
sns.set_context("poster")
sns.set_style("white")
tbl.plot(subplots=True,title=tle2, legend=True, alpha=1.0, grid=True, figsize=(11,9))
sns.set_style("dark")
tst20.plot( title=tle, legend=False, alpha=1.0, figsize=(11,9), ylim=[60,180], grid=False)
'''

# @ alerts
# quote

# ----  %LOAD2 1 - 3 ----- #
#---vectorized alert counts 1-DIM---#
def fft_vector(x):	
	'''-- rolling_window or rolling_apply
		-- return 1 value at time
	'''
	try:
		frc,fra = 0.1, 0.01
		threshold_freq=frc; 
		frequency_amplitude=fra
		
		fft_of_signal = np.fft.fft(x)
		outlier = x.max() if abs(x.max()) > abs(x.min()) else x.min()
		if np.any(np.abs(fft_of_signal[threshold_freq:]) > frequency_amplitude):
			arr = np.where(x == outlier,1,0)
			return arr[0]
		else:
			return 0
	except:
		return 0
## stat-metrics: expected no. of alerts in period -ith alert

def bayes_cp_vector(x):
	''' OFFLINE: get log-probability, Pcp, take exp. sum get p(t_i, is_changepoint)
		input:: 
			1. prior of successive[a=cp,b=cp] at t_distance
			2. likelihood_data:[s_sequence, t_distance] no changepoint 
		ONLINE: gives prob_distribution(mass) of P(t) not_cp in [1,2,...n]; n=0 is P(t) is changepoint
	'''
	# rolling_apply -> lambda x ;
	try:
		Q, P, Pcp = offcd.offline_changepoint_detection(x, partial(offcd.const_prior, l=(len(x)+1)),\
		   	offcd.gaussian_obs_log_likelihood)
		#print '** log-prob change-point t_i ',len(Pcp), '\n', Pcp
		cp = np.exp(Pcp).sum(0)
		#print 'cp\n ', cp
		return cp[0]
	except:
		return 0

#log transform
#https://pythonhosted.org/PyQt-Fit/KDE_tut.html#transformations

import pyqt_fit.bootstrap as bs
import pyqt_fit.kernel_smoothing as smooth
cimaxmin=[[]]
global cimaxmin
cimaxmin.append([])


def kern_vector2(xx ): 
	''' * NOT WINDOW-ing ; apply()
	'''
	# -- grouped object, passed in with lambda
	#xv = x[ x.iloc[['value']] ]
	x = xx['value'].values
	print 'x', x[:4], len(x)	
	
	#--grid/index
	#add/mult is element-wise with scalar.  
	#pass imaginary number to np.r_ return (0:ws) inclusive, by skipping last
	imgn = np.complex(len(x))
	grid = np.r_[0:len(x):imgn] 
	xindex = np.asarray( range( len(x) ) )
	#print 'grid lens', len(grid), len(xindex)

	#--bootstrap
	try:
		result = bs.bootstrap(smooth.LocalPolynomialKernel1D,\
		   	xindex, x, eval_points = grid, fit_kwrds = {'q': 2}, CI = (95,99))
		rmax = result.CIs[0][0,1]
		rmin = result.CIs[0][0,0]
		#print 'minmax', len(rmax), len(rmin)
		
		#--boolean array outlier
		#--vectorized lambda passes each element to expression 
		v = xx['value'].values
		c = np.where( v>rmax, 1,
				np.where( v<rmin,1,0))
		xx['krn'] = c 
		#at = ['krn'] * len(x)
		#xx['alert_type'] = at
		print 'count\n', xx['krn'].value_counts()
		#pri('xx',xx.head())
		return xx
	except Exception:
		z = np.zeros(len(x) )
		xx['krn'] = z
		return xx


def maincsv(dt=None, csvfile='thmi_alerts.csv'):
	'''write all the alerts to csv
	   thmi gets ['fft'] ['krn'] ['bcp']
	   stack to ['alert_type'] ['alert_value']
	'''
	drcty2 = '/home/solver/project/data/'
	csvf = drcty2 + csvfile

	if not os.path.exists( csvf ) :
		print '\n:::you are writing csv file ',csvfile,':::\n'
		dt.to_csv(csvf, sep='\t', encoding='utf-8')

	if os.path.exists( csvf ):
		print '\n',csvfile,' exists, not overwritten\n '
		nf = csvf[:-4] + '_TEMP_.csv'
		print '\n',nf,' written instead\n '
		dt.to_csv(nf, sep='\t', encoding='utf-8')
	
def mainfft(dt=thmi):
	print "\n::: performing fft detection :::\n"
	''' grouped only by ['variable'] '''
	fft_v = lambda x: fft_vector(x)	
	#pri('fft bfre alert', dt.head() )
	#dt['fft'] = dt.index.map(lambda x:-1)
	tg = dt.groupby(['variable'])
	gg = []
	for k,g in tg:
		# return 1 val at time
		g['fft'] = pd.rolling_apply( g['value'], 10, fft_v )
		gg.append(g)
		print k, g[['fft']][:15]
	th =pd.concat(gg)
	#pri('checking dt fft', dt.head(25))
	print 'fft alerts\n', th['fft'].value_counts()
	return th
	#pri('th', th.head(15) )
	#print 'th', th[['fft','variable']][:100]

def mainkernreg(dt=thmi):
	print "\n::: performing kernel regression detection _apply :::\n"
	'''grouped by variable, subject_id 
	#http://stackoverflow.com/questions/24272398/python-cleaning-dates-for-conversion-to-year-only-in-pandas
	http://stackoverflow.com/questions/9155478/how-to-try-except-an-illegal-matrix-operation-due-to-singularity-in-numpy?rq=1
	'''
	# -- munge
	#dt['krn'] = dt.index.map(lambda x:-1)

	# -- datetime64 issue
	pd.to_datetime( dt['realtime_x'], coerce=True )

	##cln = dt[ (dt['subject_id']==1) &( dt['variable']=='dia') ]

	# -- vectorized, but pass each group with lambda
	kr_v = lambda x: kern_vector2(x )
	cln = dt.groupby(['subject_id','variable'],as_index=False,group_keys=False).apply(kr_v)
	#pri('krn', cln.head() )
	print 'krn alerts\n', cln['krn'].value_counts()
	return cln


def mainbayes_changepoint(dt=thmi):
	#if not os.path.exists( hdf5 ): 
	print "\n::: performing bayes-point detection :::\n"


	bp_v = lambda x: bayes_cp_vector(x)
	#dt['bycp'] = dt.index.map(lambda x:-1.0)

	tg = dt.groupby(['variable'])
	gg = []
	for k,g in tg:
		print '##val check ', k,'\n', g.value[:3]
		g['bycp'] = pd.rolling_apply( g['value'], 15, bp_v )
		g['bycp'] = g['bycp'].map(lambda x: x> g['bycp'].quantile(.90) and 1 or 0)
		gg.append(g)

	th = pd.concat(gg)
	pri('bayes changepoint', th.head() )
	return th



## --------------------------------------------###
# @ data
#read write alerts --  write a new file, then change if->(0)	

#csvf = 'alert100_TEMP_.csv'
#csvf = 'alert1500mcd.csv'
##csvf = 'alert5000mcd.csv'
csvf = 'alert20Kmcd.csv'
##with new main_bycp function that discretize probs
#csvf = 'alert1Kmcd_TEMP__TEMP_.csv' 

###########################################################################
# want to get the weights, for each alogrithm
# so only run once
## write to csv, then load to dataframe
'''PIPELINE BEGIN''' #@rufus
'''
1. states 
2. flag interrupt 
3. 
'''
## @orignate thmi file

## STAGE 1 ALERTS HAS 3 DISTINCT TASKS 
@transform(start, suffix('.csv'), '.fft')
def fft_stage1(infile, outfile):
        # write thmi to file ->load to stage 1
        inf = open(infile)
        # mainfft()
        oo = open(outfile, 'w')

@transform(start, suffix('.csv'), '.krn')
def krn_stage1(infile, outfile):
        pass
@transform(start, suffix('.csv'), '.bycp')
def bycp_stage1(infile, outfile):
        pass
## STAGE 2 BOOST 1 TASK
@transform([*stage1], regex(r"(fft)(krn)(bycp)$", '.boost')
def boost_stage2(infile, outfile):
        '''
        ** module issues:
           cd ~/project/scikit-learn-master/ as
           python -m sklearn.ensemble.sdbw2.py
           have to set dd , col-headers in data file
        ** 
        ?? which one, boosting writes weights
        -> ?? boosting.py (baseclass wrapped, weight.pkl written)
        -> ??sdbw2.py (calls boosting.py, entropy(bayesblock) pickled ) 
           bwb=AdaBoostClassifier()
           fit(bwb,patData,patTarg)
        -> dont need this file -> score.py class (loads the weights, entropy -> plots)
        -> otter.py loads score.py -> gets weights
        '''
        #override the patData, patTarg
        pass

## STAGE3 BAYES 1 TASK
@transform(boost_stage2, suffix('.boost'), '.bayes')
def bayes_stage3(infile, outfile):
        pass

if(0):
	fltr = thmi[thmi['source']=='mimic'] 
	smp = fltr
	a=mainfft(dt=smp)
	b=mainkernreg(dt=smp)
	c=mainbayes_changepoint(dt=smp)

	alrt = pd.concat([smp , a['fft'],b['krn'],c['bycp']], axis=1)
	print('thmi concat alerts',alrt.head(50) )
	maincsv(dt=alrt, csvfile=csvf)
else:
	#print 'in loop'
	f = './data/' + csvf
	hdr =[ 'subject_id',  'gender',  'source',  'realtime_x',  'variable',    'value',   'index',   'timeshift',   'level_0', 'dob' ,'dod', 'realtime_y',  'fft', 'krn', 'bycp']


	alrt_data = pd.read_csv(f, names=hdr,skiprows=1, sep='\t')
	print('loaded alerts from csv', alrt_data.head())

## --------------------------------------------###

#------------------------------------------------#
# example plots of alerts fft, kernreg, bycp #
sns.set_style("darkgrid")
a19 = alrt_data[ (alrt_data.subject_id == 21) & (alrt_data.variable == 'bp') ]

def example_alertplots(xx):
    ### kernel regression
    xx = xx.tail(40)
    x = xx['value'].values[-40:]
    
    imgn = np.complex(len(x))
    grid = np.r_[0:len(x):imgn]
    xindex = np.asarray( range( len(x) ) )
    #--bootstrap
    try:
        result = bs.bootstrap(smooth.LocalPolynomialKernel1D,\
                xindex, x, eval_points = grid, fit_kwrds = {'q': 2}, CI = (95,99))
        rmax = result.CIs[0][0,1]
        rmin = result.CIs[0][0,0]
        #v = xx['value'].values
        c = np.where( x>rmax, 1,
                        np.where( x<rmin,1,0))
        xx['krn'] = c
    except Exception:
        z = np.zeros(len(x) )
        xx['krn'] = z
    #PLOT 
    sns.set_context('poster')
    fig, (a1,a2,a3,a4) = plt.subplots(4,1)

    clnc = xx.tail(40).copy()
    clnc['krnlbl'] = clnc['krn'].map(lambda x: x==0 and 'signal' or x==1 and 'outlier' )
    c = range(len(x) )
    clnc['rr'] =c
    #plt.figure()
    #sns.lmplot("rr", "value", clnc, hue="krnlbl", palette="Set1", ax=ax,fit_reg=False, ci=95); 
    sns.lmplot("rr", "value", clnc, hue="krnlbl", ax=a1,fit_reg=False, ci=95,legend_out=True);

    a1.plot(grid, result.y_fit(grid), '-', color='.2', label="Fitted curve", linewidth=1)
    a1.plot(grid, result.CIs[0][0,0], '--',color='.2', label='95% CI')
    a1.plot(grid, result.CIs[0][0,1], '--', color='.2' )

    a1.fill_between(grid, result.CIs[0][0,0], result.CIs[0][0,1], color='grey', alpha=0.25)
    #a1.legend()#.draw_frame()
    o = len( xx[xx['krn'][-40:]==1])
    n = len(x); print'n',n,o
    #a1.set_title(r'kernel regression bp : readings= %d , alerts=%d' % (n,o), fontsize=17)
    a1.set(title='Kernel Regression bp : readings= %d , alerts=%d' % (n,o)) #, fontsize=17)
    a1.set_xlim(0,n)
    #xl = plt.xlim()

    #2.  ## fft a2
    clnc['fftlbl'] = clnc['fft'].map(lambda x: x==1 and 'outlier' or 'signal' )
    #print 'fft', clnc.fftlbl[:]
    sns.lmplot("rr", "value", clnc, hue="fftlbl", ax=a2,fit_reg=False, ci=95); #, size=10);
    o = len( xx[xx['fft'][-40:]==1])
    #n = len(xx.index);
    a2.set(title='Fourier Transform: readings= %d , alerts=%d' % (n,o)) #, fontsize=17)
    a2.set_xlim(0,n)

    #3 bycp
    clnc['bycplbl'] = clnc['bycp'].map(lambda x: x==0 and 'signal' or x==1 and 'outlier' )
    data =xx.value.values
    sns.lmplot("rr", "value", clnc, hue="bycplbl", ax=a3,fit_reg=False, ci=95); #,size=10);
    #print clnc.bycplbl[:]
    o = len( xx[xx['bycp'][-40:]==1])
    #n = len(xx.index);
    a3.set(title='Bayes ChangePoint: readings= %d , alerts=%d' % (n,o)) #, fontsize=17)
    a3.set_xlim(0,n)
    #a33 = fig.add_subplot(3, 1, 2, sharex=a3)
    a1.set(xticks=[],xlabel="",ylabel='') #,title="Demographics of 2 Patient Groups")
    a2.set(xticks=[],xlabel="",ylabel='') #,title="Demographics of 2 Patient Groups")
    a3.set(xticks=[],xlabel="",ylabel='') #,title="Demographics of 2 Patient Groups")
    a4.set(xlabel="index") #,title="Demographics of 2 Patient Groups")
    a4.set(title="")
    #a1.set(ylabel="systolic") #,title="Demographics of 2 Patient Groups")
    a2.set(ylabel=r"$Systolic$") #,title="Demographics of 2 Patient Groups")
    #a3.set(ylabel="systolic") #,title="Demographics of 2 Patient Groups")
    a4.set(ylabel="probability") #,title="Demographics of 2 Patient Groups")
    a4.set_xlim(0,n)
    
    Q, P, Pcp = offcd.offline_changepoint_detection(data, partial(offcd.const_prior, l=(len(data)+1)), offcd.gaussian_obs_log_likelihood)
    ep = np.exp(Pcp).sum(0)
    nep = np.append(ep,ep[-1])
    print len(nep)
    a4.plot(nep, lw=1)
    #clnc['Pcp'] = np.exp(Pcp).sum(0)
    #print clnc.pcp.head()
    #sns.factorplot('Pcp',ax=a4,size=3)
    sns.despine(left=True, bottom=True)
    plt.subplots_adjust(hspace=2)

def rugplots():
    print 'hey'

#######################################################################################3
## rugplot, cdf of interarrival times for different 'alert_t' krn,bycp,fft
## 
'''
# -- rugplot
#(c1, c2, c3, c4, c5 ) = sns.color_palette("husl", 6)[:5]
sns.set_style("darkgrid")
g = sns.FacetGrid(d, col='alert_t', row='variable',size=1,aspect=4,\
                    palette="husl", margin_titles=False )
g.map(sns.rugplot,'tavgf',height=.2)
###print sns.axes_style()
sns.despine(left='False')
g.fig.subplots_adjust(wspace=.1, hspace=.3);
g.set_axis_labels(['time diff']);
g.set(yticks = [])
   
# -- cdf plot
c = sns.FacetGrid(d, col="alert_t",size=4, aspect=.9)       
c.map(  sns.distplot, "cumtf", kde=True, kde_kws={'cumulative':'True'}, fit=stats.expon )
#c.map( sns.kdeplot, "cumtf", cumulative=True )
c.set_axis_labels(['interarrival time']);
axes = c.axes
print 'ax ', axes
for i,ax in enumerate(axes):
    axes[0,i].set_ylim(0,1)
    axes[0,i].set_xlim(0,)


'''
#####################################################################################
def main_alerts(dt=alrt_data):
    # -- pre-example
    sns.set_style("whitegrid")
    
    # -- merge mimic and telehealth time index
    mt=dt[ dt['source']=='mimic']['timeshift']
    tt=dt[ dt['source']=='telehealth'][ 'realtime_x' ]
    dt['tidx'] = pd.concat( [mt,tt], axis=0)
    dt['timeindex'] = pd.to_datetime(dt['tidx'])
    ###print type(dt['timeindex']), dt['timeindex'].dtype
    
    # -- expanding the dataframe wide to long
    # -- melt()
    dcat = pd.concat( [dt,dt,dt], axis=0 )
    dcat['alert_t'] = ['krn']*len(dt['krn']) + ['fft']*len(dt['fft']) + ['bycp']* len(dt['bycp'])   
    dcat['alert_v'] = pd.concat( [dt['krn'] , dt['fft'] , dt['bycp']], axis=0)
    	
    dcatt = dcat[['source','subject_id','alert_t','variable','timeindex','alert_v','value']]
    d = dcatt.reset_index(drop=True).set_index(['timeindex'],drop=False ).copy()
    ###print 'd expand' , d.head(20)
    
    # -- get the interarrival time
    # -- set the iqt range
    # -- get cumsum over interarrival time
    def deltat(g):
    	try:
    		g['tavg'] = g[ g['alert_v']==1 ]['timeindex'].diff(1)
    		#print g
    		return g
    	except:
    		pass
    
    def iqt(g):
    	try:
    		g['iqt'] = g[ g['alert_v']==1 ]['value'].map(lambda x: x > g['value'].quantile(.90) and 1 or x < g['value'].quantile(.10) and 1 or 0)
    		#print 'iqt', g
    		return g
    	except (Exception, StopIteration) as e:
    		pass
    
    def cum(g):
    	try:
            #--exact[float64] conversion timedelta to seconds
            #g['tavgsec']= pd.to_timedelta(g['tavg'],unit='d')+pd.to_timedelta(0,unit='s').astype('timedelta64[s]')
            # --exact convert to float64
            g['tavg'] = g['tavg'].fillna(0)
            g['tavgf']= (pd.to_timedelta(g['tavg'],unit='d')+pd.to_timedelta(0,unit='s'))/np.timedelta64(1,'D')
            
            # --cumsum on filter rows
            g['cumt'] = g[ g['alert_v'] == 1 ]['tavg'].cumsum()
            g['cumt'] = g['cumt'].fillna(0)
            
            # float64 convert
            g['cumtf'] = (pd.to_timedelta(g['cumt'],unit='d')+pd.to_timedelta(0,unit='s'))/np.timedelta64(1,'D')
            #print 'group type', type(g['cumt'] )
            ##print g.head()
            return g
    
        except (Exception, ZeroDivisionError , StopIteration, ValueError) as e:
    		print 'cumulative error\n', e
    		pass
    
    # -- utility
    d.sort_index(axis=0, inplace=True)
    dg = d.groupby(['source','subject_id','alert_t','variable'], as_index=False, group_keys=False)
    #	pd.to_datetime(d['tavg'], format='%H:%M:%S')
    
    
    # -- set bycp threshold for probability val to alert
    def quantg(g):
        try:
            # -- vectorized if-else 
            #g['alert_v'] = np.where(g['alert_v']>g['value'].quantile(.75),1,0) 
            g['alert_v'] = np.where(g['alert_v']>.1,1,0) 
            g.drop_duplicates(inplace=True)
            #print 'group', g[:2]
            return g
    	except (Exception,StopIteration,TypeError) as e:
    	    print '**bycp error\n', e
    	    pass
    
    ## duplicate values, therefore have to reset index and drop duplicates for both groups and df original, \
    # or else update does not know which row to update new value to.
    # get_duplicates(), duplicated, drop_duplicates()
    
    #-- pre-filtered group, 
    #-- post-filter not work over multi-column, does not return unfiltered
    d.reset_index(inplace=True, drop=True)
    db=d[d['alert_t']=='bycp'].groupby(['timeindex','source','subject_id','alert_t','variable'],\
    		as_index=True,group_keys=True).apply(lambda x: quantg(x))  #.copy(deep=True) 
    db.drop_duplicates(inplace=True)
    ##print 'bycp vals', db.head()
    ##print 'bycp == 1 *** ', db[ db['alert_v']==1][:10], len(db[ db['alert_v']==1])
    
    #-- update to (unfiltered) data frame
    d1 = d.set_index(['timeindex','source','subject_id','alert_t','variable'],drop=False, inplace=False).copy()
    #print '*** bycp to_update', d[d.alert_t == 'bycp'].head()
    d1.update(db, overwrite=False)
    #print '*** bycp updated', d1[d1.alert_t == 'bycp'].head()
    ##print '*** bycp updated', d1.head(10)
    d1.reset_index(inplace=True,drop=True)
    #d.set_index(['timeindex'], drop=False, inplace=True)
    #print '*** bycp', d[d.alert_t == 'bycp'].head()
    
    # -- stupid way to split dataframe and concate alert_t
    dd = d.copy()
    ###print dd.head()
    #dd.reset_index(inplace=True)
    ddk = dd[dd['alert_t']=='krn']; ddf=dd[dd['alert_t']=='fft']
    
    d2 = pd.concat( [ddk,ddf,db], axis=0 )
    d2.set_index(['timeindex'], drop=False, inplace=True)
    ###print 'weird*** ', d2.head(), len(d2) 
    ###print 'weird*** ', d2.tail(), len(d2) 
    d2.sort_index(axis=0, inplace=True)
    ###print 'weird*** ', d2.tail(), len(d2) 
    
    #--- bycp-end ---
    
    # -- get time delta interarrival times of alerts
    d=d2.copy()
    ###print 'weird d*** ', d.head(), len(d) 
    ###print 'weird d*** ', d.tail(), len(d) 
    ###print 'weird d*** ', d.tail(), len(d) 
    
    d=d.groupby(['source','subject_id','alert_t','variable'],as_index=False,group_keys=False).apply(lambda x: deltat(x) )
    
    # -- set quartile alerts; to get FP,FN
    d=d.groupby(['source','subject_id','alert_t','variable'],as_index=False,group_keys=False).apply(lambda x: iqt(x) ) 
    
    # -- get cumulative time
    ###print '*** bycp pre 2', d[d.alert_t == 'bycp'].head()
    d = d.groupby(['source','subject_id','alert_t','variable'],as_index=False,group_keys=False).apply(lambda x: cum(x) ) 
    ###print '*** bycp2', d[d.alert_t == 'bycp'].head()
    #print d.describe() #print d.head()
    
    # -- filter out timedeltas eq 0 
    aa=pd.to_timedelta('00:00:00')
    ###print '*** bycp tod2', d[d.alert_t == 'bycp'].head()
    d = d[ pd.to_datetime( d['cumt'] ) - pd.to_timedelta('00:00:00') > pd.to_timedelta('00:00:00') ]
    ###print '*** bycp tod++', d[d.alert_t == 'bycp'].head()
    
    # fp/fn vs iqt boxplot
    # -- 1 box per column -> use pivottable
    def fp(x):
    	iqt = x['iqt']
    	alv = x['alert_v']
    	x['fpfn'] = x.apply(lambda x: x['iqt']==0 and x['alert_v']==0 and 'TN'
    								or x['iqt']==1 and x['alert_v']==1 and 'TP' 
    								or x['iqt']==0 and x['alert_v']==1 and 'FP'
    								or x['iqt']==1 and x['alert_v']==0 and 'FN',
    								axis=1 )
    	return x
    
    d = d.groupby(['source','subject_id','alert_t','variable'],as_index=False,group_keys=False).apply(lambda g: fp(g) )
    return d

d = main_alerts()
print('d', d.head())
# ----  %LOAD2 2 - 3 ----- #
###########################################################################
# box plot of noise 'fpfn'(tp,fn) of alert_t (fft, bycp, krn)
# get count over group object using agg (can set count col name as tuple to agg function
# 'fpfn' is set in function that check against iqt
# groupu over alert_t and fpfn
#def gcnt(g):
#    g['cnt'] = g.count()
#    return g
#dmain = dmain.groupby(['subject_id','alert_t','fpfn'], as_index=False, group_keys=False).apply(lambda x: gcnt(x))
'''
sns.set_style("darkgrid")

a19 = alrt_data[ (alrt_data.subject_id == 21) & (alrt_data.variable == 'bp') ]

example_alertplots(a19)

#d = main_alerts()

#a1 = sns.boxplot(cp, ax=a5)
sns.set_style("whitegrid")
#a5=sns.factorplot('fpfn',y='iqt', col='alert_t',hue='fpfn', kind='box',\
#data=dmain, dropna=True, aspect=1, size=3.5)
a5=sns.factorplot('fpfn',y='iqt', hue='alert_t', kind='point',\
data=dmain, dropna=True, dodge=0.2) #, aspect=1, size=3.5)
a5.set(title='Accuracy of Alert Types', ylabel='count', xlabel='')

'''
dmain = d.copy(deep=True)
dmain = dmain.groupby(['alert_t','fpfn']).agg('count')
print dmain
dmain.reset_index(inplace=True)
print dmain.head()

############################################################################################################
### pmf plots of interarrival time
# cdf is cut into thirds of overall interarrival time
# segments of each variable['bp','wht', etc ] are then cut from those bins
# cdf plot of each bin is plot along three axis sharing the y-axis
# the x-axis is non-overlapping, and marks off the bins
# group object : keys are alert_t and variable .. used by kdeplot as tuple; values are the tavgf (interarrival time) and the bins
###############################################################################
'''
for at in sorted(alrt): #['krn' 'bycp' 'fft']
    lbl=('first','second','third')
    lb = ['first','second','third']
    f = ''.join( ['f', str(i)] )
    f,lbl = plt.subplots(1,3, sharey=True,figsize=(10,2.4) )
    for k,v in alerttype_dict.items():
        if k[0] == at: #'krn'
            dv=v[0]; bins=v[1];
            #print (bins!='third').all()
            #print dv[bins=='first'].values
            #break
            for i,ll in enumerate(lb): 
                #print 'LBL ', ll
                #print lb
                if (bins != ll).all(): #.any(axis=1):
                    #print('$$', (bins != ll).all() )
                    break
                else:
                    #print '** ', ll
                    #print dv[bins==ll].values 
                    a=sns.kdeplot( dv[ bins==ll ].values, ax=lbl[i], label=k[1],\
                                shade=True)
                    a.set(title= r'$Interarrival Distribution \ %s $' % (at))
        else:
            pass
plt.autoscale(tight=True)
'''
dd = d.copy(deep=True)
pri('d', d.head())
#non overlapping groupby
q_nonoverlap = dd.groupby(['alert_t'])['tavgf']
binstotal= { k:v.quantile([0.33,0.66]).values for k,v in q_nonoverlap }
print binstotal
qg = dd.groupby(['alert_t','variable'])['tavgf']
#loop over group-object of 'alert_t','variable'
alerttype_dict = defaultdict(list)
for k,v in qg:
    quantl = binstotal[k[0]]
    vmax = v.values.max()
    if quantl[1] > vmax:
        vmax = quantl[1] + 1
    bins = [0,quantl[0],quantl[1],vmax]
    #cut returns labels for each index over bin labels
    dcut = pd.cut(v, bins, labels=['first','second','third'] )
    print k[0], k[1]
    alerttype_dict[ (k[0],k[1]) ] = [v,dcut]
print alerttype_dict
lbl=['first','second','third'] 
vtl = dd['variable'].unique()
alrt = dd['alert_t'].unique()
print alrt

###################################################################################
### busiest time of kde overlaps
# do a count over the interarrival times for each 'alrt' and label
# dont need to separate out the 'variable'
# need the label separate out
# take max of that

#take kde plot, get vals
# get max of kde plot over kdes for each third-interval
# plot as dashed vline -- busiest 

#multi-index dict(list)
lbl = ['first','second','third']
lad = defaultdict(list)
for a in sorted(alrt):
    for l in lbl: 
        for k,v in alerttype_dict.items():
            if k[0] == a:
                d=v[0]; b=v[1];
                lad[(a,l)].append( d[ b==l ])

#create histogram, get max
def dal(d,a,l):
    #get list, iterate
    alst = d[a,l]
    vv = []
    for v in alst:
        vv.extend(v)
    #get histo
    hst = np.histogram(vv, bins=10)
    mx = np.max(hst[0]); amx = np.argmax( hst[0] )
    hi = hst[1][amx]
    return hi

vh = [ dal(lad, a, l) for a in sorted(alrt) for l in lbl]
print vh

#plot vlines on axis
#get_ylim()

#------------
#freq -> variance; between and within

def cutbins(g):
    #get alert_type, bin values
    ky = g['alert_t'].unique()   
    quantl = binstotal[ky[0]]
    vmax = g['tavgf'].max()
    if quantl[1] >= vmax:
        vmax = quantl[1] + 1
    bins = [0,quantl[0],quantl[1],vmax]
    #cut returns labels for each index 
    g['dcut'] = pd.cut(g['tavgf'], bins, labels=['first','second','third']) 
    print g.head()
    return g

ddfv = d.copy(deep=True)
# get bin values, non-overlapping
q_nonoverlap = ddfv.groupby(['alert_t'])['tavgf']
binstotal= { k:v.quantile([0.33,0.66]).values for k,v in q_nonoverlap }

#pre filter groupby
# - prefilter, add labels
ddf3 = ddfv.reset_index(inplace=False,drop=True)#.copy(deep=True )
ddf3 = ddf3[ddf3['alert_v']==1].groupby( ['source','subject_id','alert_t','variable'],as_index=False,group_keys=False).apply( lambda x: cutbins(x) )

dm = pd.merge(ddfv,ddf3,how='outer')
print dm.head()

#zscore pivot -> mean/zscore -> filter alert_v==1 only
zscore = lambda x: np.absolute((x- x.mean())/x.std() )
dm['zscore'] = dm.groupby(['source','subject_id','alert_t','variable'],as_index=False,group_keys=False)['value'].transform(zscore)
dmz= dm[['dcut','zscore','alert_t']]
dmz=dmz.dropna()
print dmz

dzs = dzs.pivot_table('value',rows='tavgf', cols='alert_t', aggfunc=lambda x: zscore(x) ) 
print dzs
'''
f, (ax1, ax2) = plt.subplots(1, 2)
a1= sns.pointplot("dcut", "zscore", hue="alert_t", data=dmz, ax=ax1, dodge=0.2, ci=15)
a2=sns.barplot("alert_t", "zscore", "dcut", data=dmz, ax=ax2, ci=15)
a1.set(title='variance vs time_period')
'''

#violin plot values->distribution over all subjects, mean
##dm2 = dm.copy(deep=True)
##dvln = dm2[['variable','alert_t','value','dcut']]
##dvg = dvln.groupby(['variable','alert_t'])
##    f,lbl = plt.subplots(1,3, sharey=True,figsize=(10,2.4) )
##vln = {}
##for k,v in dvg:
##    print v[['value','dcut']]
##    print k
##    vln[k] = v[['dcut','value']] #.dropna()
#--dcut
dm2 = dm.copy(deep=True)
dvln = dm2[['variable','alert_t','value','dcut']]
dvg = dvln.groupby(['variable','dcut'])
vln = {}
for k,v in dvg:
    print v[['value','alert_t']]
    print k
    vln[k] = v[['alert_t','value']] #.dropna()

'''
# violin plots 1,2,3
vr = sorted( dvln.variable.unique() )
at = sorted( dvln.dcut.dropna().unique() ) 
al = ('a0','a1','a2')
for v in vr:
    f = ''.join(['f',str(i)])
    f,al = plt.subplots(1,3, sharey=False,figsize=(10,2.4) )
    for i,a in enumerate(at):
        try:
            d = pd.DataFrame(vln[v,a]).dropna()
            ab = sns.violinplot(d.value, d.alert_t, ax=al[i] )
            ab.set_ylabel(v)
            ab.set_xlabel(a)
        except: #e as 'ValueError':
            pass

'''

#cohens d-stat effect size between periods
dc = dm.copy(deep =True)
print dc.head()

#coefficient of variation exponential/poisson

#fit distribution
#toy
tr = np.random.randn(2,5)
fd = dict(a=tr[0] , b=tr[1] )
atst = ['a']*5
print atst


def CohenEffectSize(group1, group2):
    diff = group1.mean() - group2.mean()

    var1 = group1.var()
    var2 = group2.var()
    n1, n2 = len(group1), len(group2)

    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    d = diff / math.sqrt(pooled_var)
    return d




       
#z-score
def zscr(g):
    #zscore
    g['zscr'] = ss.zscore(g['value'],ddof=1)
    print g['zscr'].head()
    return g
#get zscore over all points
dzf = dzf.reset_index().groupby(['alert_t','dcut']).apply(lambda x: zscr(x) )
print dzs.head()
f, (ax1, ax2) = plt.subplots(1, 2)
sns.pointplot("dcut", "zscr", "alert_t", data=dz, ax=ax1)





t_period
sns.pointplot("dcut","var","alert_t")



#donito


#ruffus pipeline
'''
write dataframe to fft.csv, krn.csv, bayes.csv
ruffus- checkpointing
fft.csv,krn.csv, bycp.csv -> [boost] -> fft_boost.csv, krn_boost.csv, bycp_boost.csv -> [bayes] -> fft_bayes.csv, krn_bayes.csv, bycp_boost.csv
'''


#-------------------------------------------------#
# boosting #

# wts  I/C  H/E  variable

def bin_weight(dt=wt):
        #1 bayes-block
        #wt_bayes = bayes_block(wt) 

        #2
        lbl = ['I', 'II', 'III', 'IV']
        lbld = { i:l for l,i in enumerate(lbl) }
        b = [lbld[w] for w in len(wt_bayes) ]
        cut_col = pd.cut(wt, bins=wt_bayes, labels=b)
        dd = dict(wts=wt, cuts=cut_col)

        #3
        d = pd.DataFrame(dd)
        df = d[ d['cuts' == 'I'] || d['cuts' =='II'] || d[cuts=='III'] ]

        #4
        gs = plt.GridSpec(3,1)
        plt.subplot(gs[:2])
        ax1 = sns.pointplot( df["cuts"], df["wts"], label='d', color='0.3')
        ax1.set(xticks = [], xlabel = '')
        plt.subplot(gs[-1])
        ax2 = sns.barplot( df['cuts'], color='0.5')
        ax1.set_xlim(ax2.get_xlim() )
        sns.despine(left=True, bottom=True)


def hard_easy(d):
        #1
        he = lambda x: x>threshold and 'H' or 'E'
        d['HE'] = d[wts].apply(he)
        #2 heatmap
        dh = d[ d['wts','variable'] ]
        dh = dh.unstack('variable')
        sns.heatmap(dh, ax=ax2)
        #3 plot
        f, (ax1,ax2) = plt.subplots(1,2)
        sns.pointplot('IC', data=d, ax=ax1)

def INROW123(d):
        #1 window for 1-2-3
        dc = range(10)
        d4 = d[dc].stack(name= '123ROW')
        #2
        gs = plt.Gridspec(3,2)
        plt.subplot(gs[:2])
        ax1 = sns.pointplot(dopt['123ROW'], dopt['he'],
                label = 'd', color='0.3')
        ax1.set(xticks=[], xlabel='')
        plt.subplot(gs[-2] )
        ax2.sns.barplot(dopt['123ROW'], color='0.05')
        ax1.set_xlim(ax2.get_xlim() )
        plt.subplot( gs[-1] )
        sns.despine(lefrt=True, bottom=True)
        ax3 = sns.pointplot(['win-len'], data=d4, color='blue')

#bayes -- hierarchical model --likelihood function, estimation(german tank)
def bayes_hmdl(pool, ):
        '''
       -- graph:
        how?picture of nurse MIMIC
        kruske diagram or bayes net

       1.linear regression model
       2.hierarchical model 3 states (mortality, gender, alert_type)
       featurematrix(mortal,gender,geo) + alert_type
       bayes-net
       3.streak (monty_hall) 
       3a.sequential sprt wald-boost
       4.hashing kernel
       5.percolation
       6.intprogram_constraints
       7.expert_priors(cardiologist, mturk, watson)
       '''
       #alerthard_i,p = alpha + beta*interarrival + err
        


        


def bayes_decision: pass
        '''
        2.simulation
        2aa. norvig simulation model
        2a. generative, explain the data
        2b. mixture model- chinese buffet
        2c. ternary tree
        2d. resolution limit problem
        2e. darknet
        3.estimation (forward)
        number of alerts, time to alert, 
        4.visualization
        1.likelihood (inverse)
        2.causation
        '''
        












                        














