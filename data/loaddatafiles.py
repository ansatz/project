import os,csv
from collections import OrderedDict

class PtID(object):
	def __init__(self, id):
		self.id = id

# csv -> csv
#def sort_k_time(dir):
#	#find all vitals near each other for 1 patient
#	#/expired/female or /expired/male
#	   # cant treat path like string: have to os.path joindirname = '/expired/' + dir  #/male or female/
#	#dd=[('SYS', 'float64'),('DIA','float64'),('HR1','float64'),('OX','float64'),('HR2','float64'),('WHT','float64'),('Label',int)]
#	HDR = ['TIME','SYS','DIA','HR1','OX','HR2','WHT','Label','PTNT','SEX']
#	chartID={'wht': { 'daily':763, 'present_lb': 3581, 'na': 1148 } , 
#			 'bp': 455 ,
#			 'ox': 646 ,
#			 'hr': [211,618],
#
#
#	fname = [ os.path.join(dir,name) for name in os.listdir(dirname) ]  #list file names
#	
#	for fs in fname: 
#		with open( fs, 'rb') as csvfile:
#		#load all files into some csvreader object, can get each column by index
#			rd = csv.reader(csvfile) 
#			for row in fname:
			




def allcsv(*args):
	# one-liner gets only files in dir, 
	# -- dont use '/', pass in *args
	cwd = os.getcwd() ; dri = os.path.join(*args) ; pathy = os.path.abspath(dri)

	#no sub-dirs, load all the csv files in folder
	filenames = next(os.walk(pathy))[2]   
	allcsv = [ f for f in filenames if('.csv' in f) ]

	#list, hold all rows to write
	allc = []; pt=[]
	fwr = os.path.join(pathy,'all.csv')

	#go through dir-files
	for i,drfl in enumerate(allcsv):
		# -- add path
		pfl = os.path.join(pathy,drfl)
		unq=[]
		pt.append([])

		# - read file row by row (1000 lines)
		with open(pfl,'r') as fl:
			rd = csv.reader(fl)
			
			# - get unique
			print i, pfl
			for row in rd:
				pt[i].append(row[1])
			unq = ( OrderedDict.fromkeys(pt[i]).keys() )  
			# -- another one-liner for unique_items = [unicode(ch) for ch in sorted(set(s), key=s.index)]
			# -- or can also sort keys .. sorted( somedict.keys() )
			
			# - write pt data to list	 
			# -- only files with 2 patients -- discard 2nd
			if len(unq) == 2:
				#rd2 = csv.reader(fl)
				fl.seek(0)
				for rowr in rd:
					if rowr[1] == unq[0]:
						allc.append(rowr)	

	# - write to file
	with open(fwr ,'w') as fall:
		allwrt = csv.writer( fall )
		allwrt.writerows( allc )

	print 'lines of data', len(allc)

#allcsv('expired','male')



#panda load csv -> sort k-time -> graph summary,box-plot,etc
#all data from sql, so in itemid in one column
#df.sort_index(axis=1)
import pandas as pd
import matplotlib.pyplot as plt


def sortK( ):
	hdr =['index','subject_id','sex','dob','dod','hospital_expire_flg', 'itemid', 
			'description', 'charttime', 'realtime',
			'value1num','value1uom', 'value2num','value2uom']

	df = pd.read_csv("expired/male/all.csv", names=hdr )
	#format the realtime
	df['realtime'] = df.realtime.map(lambda x: pd.datetools.parse(x).time())
	#set index
	#df.set_index(['subject_id'], inplace=True)

	#qp = df['value1num'].apply(int)
	qp = df[['value1num','itemid']]
	print qp[:5]
	print qp

	vitals = {}
	pg = df.groupby(['subject_id','realtime'])
	#sort by default
#	print pg.groups
	for k,v in pg:
		print v['subject_id','itemid','realtime']



sortK()

#group by range
grouprange = lambda x: if x == x.next or x.next

df['group'] = df['charttime'].apply(lambda x: 

	

pg = pt.cut(['charttime', bins= pt['charttime'].map(grouprange) , labels=False)
#set indices of patient,itemid
#unstack turn itemid into columns
pt['times'] = pd.cut(['charttime', bins=pt['charttime'] )
	

def unq1(n): return lambda: n+1 
u = unq1()

def uniquegroups(x):
	grp = 1
	dct = []
	def unq():
		if x not in dct:
			dct.append(x)
			return grp
		else: 
			dct = []
			grp += 1
			return
	return unq

pt.groupby('charttime','realtime').map(uniquegroups)
pt['group'] = pt.map(uniquegroups)


#dt['group'] = dt['itemid'].map(lambda x: )
#def f(dataframe=dt):
#	# order by times
#	
#
#
#	# select columns, get unique set
#	dt['groups'] = dt['itemid'].unique
#
#
#	# d['groups'] =  df.f
#

#qp = qp.astype('float')
#qp.plot()
#plt.figure()
#pd.options.display.mpl_style = 'default'
#df.plot(kind='bar')
#ts = pd.Series(randn(1000), index=date_range('1/1/2000', periods=1000))
#ts=np.exp(ts.cumsum())
#ts.plot(logy=True)
#plt.legend(loc='best')
#	

#class Load(object):
#	def __init__(self, dir):
#		self.dir = dir
#
#	DTASRC = 


"""
1. input
#load data files from

tele_raw_dta = np.recfromcsv("/home/solver/data/raw-labeled-th/all.csv", dtype=dd)
mimic_raw = np.recfromcsv("/home/solver/Desktop/data/raw-labled-mimic/all.csv", dtype=dd)

2. output
#cross-validate
gender, expired

score.py object

boost -> creates 4 files  weight.pkl incorrect.pkl entroypbasic.pkl datahdrlbl.pkl
W the boosting weights
E entropy
I incorrect =1 correct =0
D header [vitals[the header], alerts[patTarg], readings[patData] ]

in scikit-learn-master/ when boosting.py is run
1. create new dir for each run
2.#load data from paths is hard set 
dd = 
`
	# load data from paths.
	dd=[('SYS', 'float64'),('DIA','float64'),('HR1','float64'),('OX','float64'),('HR2','float64'),('WHT','float64'),('Label',int)]
	tele_raw_dta = np.recfromcsv("/home/solver/data/raw-labeled-th/all.csv", dtype=dd)
	mimic_raw = np.recfromcsv("/home/solver/Desktop/data/raw-labled-mimic/all.csv", dtype=dd)
	
	x = [map(parseNum, line) for line in csv.reader(open("/home/solver/Desktop/data/raw-labled-mimic/all.csv"))]
	bst = np.asarray(x)
	bstRows = bst.shape[0]
	bstCol = bst.shape[1]; #print bstCol
	patData = bst[:, 0:bstCol-1].copy()  #everything but last column(labels)
	patTarg = bst[:, bstCol-1].copy()
	print("l262, patdat pattar", patData.shape, patTarg.shape)
`

####
csv -> only data ((no headers)) and label 1,-1
100.3,62.3,79.9,489,79,13.3,1

the new mimic from sql query is: 
"1","5077","F","3137-11-05 00:00:00.0","3215-12-05 00:00:00.0","Y","211","(null)","3210-08-07 08:30:00 -0500","3210-08-07 08:29:00 -0500","97.0","BPM","(null)","(null)"

header: 
select ce.subject_id, 
       pt.sex, pt.dob, pt.dod, pt.hospital_expire_flg, ce.itemid, 
	          ci.description, charttime, realtime, 
			         value1num,value1uom, value2num,value2uom

telehealth header:
{ "TIME":"2008-12-25 22:56","SYS":158,"DIA":77,"HR1":70,"OX":97,"HR2":68,"WHT":173.60,"PTNT":0 },


ce.itemid in (455,1148,618,115,211,3581)

[bp(dia,sys) , wht , sp02, hr ]
>> ce.itemid in ( (455) and (763 or 3581) and (646) and (211 or 618) ) 

present weight: 3581 (lb) and 1148
heart rate: 211
SpO2: 646
NPB: 455
bpm heart-rate: 618
646: spo2
weight: 763 daily weight


need 115 wrong! it is capillary refill
select subject_id
from chartevents
group by subject_id
having (max(case when itemid in (763,3581) then 1 else 0 end) +
        max(case when itemid in (455) then 1 else 0 end) +
        max(case when itemid in (211,618) then 1 else 0 end)
       ) >= 2

select subject_id
from chartevents
group by subject_id
having sum(case when itemid in (763,3581) then 1 else 0 end) > 0 and
       sum(case when itemid in (455) then 1 else 0 end) > 0and
       sum(case when itemid in (211,618) then 1 else 0 end) > 0

explain plan for
select subject_id
from chartevents
group by subject_id
having sum(case when itemid in (763,3581) then 1 else 0 end) > 0 and
       sum(case when itemid in (455) then 1 else 0 end) > 0 and
       sum(case when itemid in (211,618) then 1 else 0 end) > 0

select t1.subject_id
from chartevents t1 
inner join chartevents t2 on t1.subject_id = t2.subject_id
inner join chartevents t3 on t1.subject_id = t3.subject_id
where 
  t1.itemid in (763,3581)  
  and t2.itemid = 455
  and t3.itemid in (211,618)
"""












