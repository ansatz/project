########################################################
'''
ipython notebook to deal with pandas, seaborn, and bokeh stack
'''
#######################################################
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from collections import defaultdict
#ipython notebook --pylab inline

from scipy import stats, optimize
import seaborn as sns
sns.set(palette='Set2')

########################################################
class Grapes(object):
	peas = 0
	pod=[[]]

	def sum(self):
		Grapes.peas +=1

	def applesin(self,g):
		Grapes.peas += 1
		Grapes.pod.append([])
		Grapes.pod[-1].append(g)

	def applesout(self,g):
		Grapes.pod[-1].append(g)

	def bunches(self,g):
		'''g is row value from itemid'''
		for i in self.pod[-1]:
			if (g == i):
				self.applesin(g)
				return Grapes.peas
		else:			
			self.applesout(g)
			return Grapes.peas

def getweekday(dataframe):
		dt = dataframe['realtime']
		wkd = lambda x: somefunc(x) 
		dt.map(wkd )		
pt['weekday'] = pt['my_dt'].apply(lambda x: x.weekday())

###########################################################
''' regression data
-- boosting scatterplot with decision boundary
-- weights of boosting http://stanford.edu/~mwaskom/software/seaborn/examples/many_facets.html#many-facets
'''




''' categorical data
-- factorplot week
-- hard_easy_correct_incorrect_alerts
'''

def facetedHEIC():
	# row=hard
	
		
	
	# tips = sns.load_dataset("tips")
	# g = sns.FacetGrid(tips, row="sex", col="time", margin_titles=True)
	# bins = np.linspace(0, 60, 13)
	# g.map(plt.hist, "total_bill", color="steelblue", bins=bins, lw=0)
	









##########################################################
def icuToboost():
	'''load icu data into pandas
		pass to boosting function
	'''
	pass
############################################################################	
def testplt():
    def random_walk(n, start=0, p_inc=.2):
        return start + np.cumsum(np.random.uniform(size=n) < p_inc)
    
    starts = np.random.choice(range(4), 10)
    probs = [.1, .3, .5]
    walks = np.dstack([[random_walk(15, s, p) for s in starts] for p in probs])
    sns.tsplot(walks)
    
testplt()
############################################################################	
#df is long-form no index
def timeaverageplots(df):
	'''_dummy time index'''
	#df["_dummy"]="_"

	''' long form table, no index '''
	#ax = df['value1num'].groupby(["itemid","realtime"]).mean().unstack("itemid").plot()	

	''' time index '''
	#x = np.arange(len(df.realtime.unique()))
	#palette = sns.color_palette()

	'''confidence intervals'''
	#for cond, cond_df in df.groupby("itemid"):
    #	low = cond_df.groupby("realtime").value.apply(np.percentile, 25)
    #	high = cond_df.groupby("realtime").value.apply(np.percentile, 75)
    #	ax.fill_between(x, low, high, alpha=.2, color=palette.pop(0))	
	
	'''resample to align timeseries'''
	'''shift so weekday line_up'''
	'''group'''
	'''plot over a max_good range'''

timeaverageplots(pt_raw)

#plot average confidence intervals
def pltavgcis(df, item='itemid', subject='subject_id', value='value1num'):
	'''split apply'''
	sid = df.groupby([item,subject])[value].describe()
	'''mean, ci, '''
	#gavg = lambda g: mean(g)
	#sid.apply(gavg)
	sid.mean()
	'''fill between, plot'''

	return sid

pltavgcis(pt_raw)

#plot average confidence intervals
def pltavgcis(df, item='itemid', subject='subject_id', value='value1num'):
    '''split apply'''
    sid = df.groupby([subject,item])[value]#.describe()
    '''mean, ci, '''
    #gavg = lambda g: mean(g)
    #sid.apply(gavg)
    #sid.mean()
    '''fill between, plot'''
    
    return sid.describe()

pltavgcis(pt_raw)

##
#>>>>>plot average confidence intervals
def pltavgcis(df, item='itemid', subject='subject_id', sex='sex', value='value1num'):
    '''split items'''
    sid = df.groupby([sex,item])[value] #[value].describe()
    
    '''apply quartiles, mean '''
	t5 = lambda x: np.percentile(x,25)
	t7 = lambda x: np.percentile(x,75)

	functions = ['mean', 't5', 't7']
   	sid = sid.agg(functions)

    '''sublplot of each itemid'''
    #sid = sid.plot(subplots=True)
    
    
    '''fill between, plot'''
    
    
    
    ''' return object ''' 
    return sid

pltavgcis(pt_raw)




##############################################################################
def alertNumber():
	'''histogram of alerts correct incorrect overlap
		http://www.stanford.edu/~mwaskom/software/seaborn/tutorial/plotting_distributions.html#basic-visualization-with-histograms'''





def jitter():
	''' x-axis is times 
		multiplots of the data	
		
	'''
	pass		

#panel of summary data
def summarypanel():
	'''bp, ox, hr, wt'''
	sns.factorplot(\5	



#input csv
hdr =['index','subject_id','sex','dob','dod','hospital_expire_flg', \
         'itemid', 'description', 'charttime', 'realtime', 'value1num',\
         'value1uom', 'value2num','value2uom']
#ICU
pt = pd.read_csv("./expired/male/all.csv", encoding='latin1', names=hdr, \
              sep=',', parse_dates=True, dayfirst=True, format='/%y/%m/%d /%H/%M/%S' )
pt[:5]

#telehealth


#long to wide
ptd = pt.drop_duplicates(['realtime'])
pt2 = ptd.pivot('realtime','itemid','value1num')
pt2[:10]







'''
#merge sort
pt.set_index('subject_id')
pt.sort('realtime', ascending=True, inplace=True)

#group by range
ugr = Grapes()
print 'UGR ', ugr
pt['group'] = pt.apply(lambda row: ugr.bunches(row['itemid']), axis=1 )
pt.head()

#unstack groups
pt.unstack('group')

#reshape
#pt[:10]
#pt[:10].plot()
#pt2 = pt.unstack('itemid')
#pt2[:10]

#table = pd.pivot_table(pt, values=['value1num','value1uom'], rows=['realtime'], cols='itemid')
#table[:10]

#long to wide
ptd = pt.drop_duplicates(['realtime'])
pt2 = ptd.pivot('realtime','itemid','value1num')
ptd[:10]
pt2[:10]
#pt.stack('itemid')
#pt[:10]
#pt.unstack()
#pt[:10]
#group by range
#pt.groupby('realtime')
#ugr = Grapes()
#print 'UGR ', ugr
#print ugr.bunches(2)
#pt['group'] = pt.apply(lambda row: ugr.bunches(row['itemid']), axis=1 )
#pt2 = pt.stack('group').unstack('itemid')
#pt2[:10]
#table = pd.pivot_table(pt, values=['value1num','value1uom'], rows=['group'], cols='itemid')
#table[:10]



#pt['group'] = pt['itemid'].map(lambda x: bunches(x) )
#pt['realtime'] = pd.to_datetime(pt['realtime'])   

#pt.sort('realtime', ascending=True, inplace=True)
#gp = pt.groupby('realtime')#.unstack('itemid')
#gp[:10]
#for k,v in gp:
#    print(k)
#    print( v['itemid','value1num'] )
#pt[['realtime','itemid','value1num']][:10]
#pt[:100]
#pt['realtime'] = pt.realtime.map(lambda x: pd.datetools.parse(x).time())
#pt['realtime'] = pt.realtime.map(lambda x: pd.datetools.to_datetime(x))

#ugr = Grapes()
#print 'UGR ', ugr
#print ugr.bunches(2)
#pt['group'] = pt.apply(lambda row: ugr.bunches(row['itemid']), axis=1 )


#pt.set_index('itemid', inplace=True)
#pt['value1num'].unstack('itemid')
#pt[['realtime','itemid','value1num']][:10]
#pt = pt.reindex(columns=labels[::-1], index=df.index[::-1])
#pt[['realtime','itemid','value1num']][-10:]


#pt2 = pt.pivot(index='realtime', columns='itemid', values='value1num')
#pt2 = pt.unstack('itemid')

#panel = pt.set_index(['itemid']).sortlevel(0).to_panel() #.first()
#panel= pt.stack('realtime')
#panel[-10:]

#panel.unstack('realtime')
#pt[['realtime','group','itemid']][-10:]
#pt.head()
#pt.unstack('itemid').head()
#pt[['realtime','group','itemid']].unstack('itemid')[-10:]




'''
