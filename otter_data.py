#################################################################################################
'''
This is the main, cleaned up file to load data.
csv to pandas

In effect study design is done here, and question to answer put here.

--------------------------------------------------------------------
# three types of variables
(p3 of rstat/MedicalStatistics -A guide to data analysis)
outcome variables/dependent/ y-axis, col
intervening variables (secondry alternative outcome var, y-axis)
explanatory variable (iv's, riskfactors, exposure var, predictors, x-axis rows)

depedent on study design, eg in case-control where disease status is seletion criteria, expVar is +/- disease, outcomeVar is exposure.
in obs/exp studies(clinical,cross-sectional,cohort,panel) disease is the outcome, exposure is explanatory...
** can use the case-control in disease modeling **

can be categorical/order/non-ordered or continous ===>>> for classification for analysis dec

## pre analyze steps, p6 of rstat/MedicalStatistics##
get min-max and range of each variable
get frequeny analysis for categorical var
use box-plots, histograms, other test to determine normality
id/deal with missing values and outliers ***

#outliers as summary pg15 of rstat/MedicalStatistics
univariate : 99% within 3std

and multivariate
(uni for one variable, uni for another) cook's distance, mahalonobis
#tests p16 rstat/MedicalStat
binary , once:incidence/prevalance/ci twice:chi-square
continous, once:normality test-ks twice:pairedT >:ANOVA,glm

#n=30> for normality

# share data by resampling and not sharing the original
http://nbviewer.ipython.org/gist/aflaxman/6871948
'''

#################################################################################################
#ipython notebook --pylab inline
#% load otter
#% matplotlib qt

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from collections import defaultdict
from scipy import stats, optimize

import seaborn as sns
sns.set(palette='Set2')
sns.set_context('talk') #paper,notebook,poster

from score import *
import numpy as np
import matplotlib.pyplot as plt

# - 1. mimic
def m2d():
	hdr =['index','subject_id','sex','dob','dod','hospital_expire_flg', \
	         'itemid', 'description', 'charttime', 'realtime', 'value1num',\
	          'value1uom', 'value2num','value2uom']
	
	#th_data['source'] = th_data.apply( th )
	#pt_data = pd.read_csv("./data/expired/male/all.csv",encoding='latin1',names=hdr,sep=',', index_col='realtime', parse_dates=True, dayfirst=False)
	pt_data = pd.read_csv("./data/mimic2v26_1_6.csv",encoding='utf-8',names=hdr,sep='\t', index_col='realtime', parse_dates=True, dayfirst=False)
	#pt_data.head(2)
	return pt_data

#exit(0)
# - 2. telehealth
#TODO add time index, sex, geography
##hdr1 = ['bp','hr1','ox','hr2','wht','lbl' ]
##pt1_data = pd.read_csv("./data/thdta/raw-labeled-th/all.csv",encoding='latin1', names=hdr1, sep=',') #, index_col='realtime', parse_dates=True, dayfirst=False)
##print pt1_data.head()
##
####################################################################################################
### --context switch => output will be below
##context_data = pt_data
##
###################################################################################################
### - clean data --(null) na
##def cleanr(x):
##    if x==r"(null)":
##        return "NaN"
##    else:
##        return x
##context_data = context_data.applymap(lambda x: cleanr(x))
##any(context_data is "null")
##context_data = context_data.dropna()
##any( pd.isnull(context_data) )
##
### -- str to floats
##context_data['val1'] = context_data['value1num'].map(lambda x: float(x))
###pt.dtypes
##
### -- map vitals
##context_data["vitals"] = context_data.itemid.map({211: "hr1", 455: "sys", 618: "hr2", 646: 'ox', 763: 'wht'})
##
### -- weekday --only works if data is index
###wk = context_data.copy(deep=True)
###wk = wk[['vitals','val1']]
###wk['weekday'] = wk.index.weekday
###wk["week_day"] = wk.weekday.map({0: "Mon", 1:'Tu', 2:'Wed', 3:'Th', 4:'Fr', 5:'Sa', 6:'Su' })
##
##context_data['weekday'] = context_data.index.weekday
##context_data["week_day"] = context_data.weekday.map({0: "Mon", 1:'Tu', 2:'Wed', 3:'Th', 4:'Fr', 5:'Sa', 6:'Su' })
##
##
########################################################################################################
### -- output => context_data index='realtime', 'week_day', 'vitals' 
##print( context_data.head(3) )
#########################################################################################################
