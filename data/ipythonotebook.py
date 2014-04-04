import pandas as pd
import matplotlib.pyplot as plt
#pd.options.display.mpl_style = 'default'
from matplotlib import rcParams
from collections import defaultdict

hdr =['index','subject_id','sex','dob','dod','hospital_expire_flg', 'itemid', 'description', 'charttime', 'realtime', 'value1num','value1uom', 'value2num','value2uom']

pt = pd.read_csv("./expired/male/all.csv", encoding='latin1', names=hdr, sep=',', parse_dates=True,dayfirst=True)
pt[:5]

#pt.set_index(['subject_id','charttime'], inplace=True)
pt.set_index('subject_id')

global ctr
global gg
#gg[0].append([])
ctr = 0; 
gg=[[]]
print 'hi', len(gg)
gg[-1].append('a')
gg[-1].append('b')
gg.append([]); gg[-1].append('a');
gg[-1].append('b')

def increase():
	global ctr
	ctr += 1;

def add1(ct=[]):
	def add(n,a=1,ct=ct):
		ct.append(n)
		sm=0
		for i in ct:
			sm+=i
		return n+a
	return add
a=add1()
print '3', a(2)
print '5', a(2)

def uniquegroupsA():
	#g is row value from itemid
	#print('row value is',g)
	ctr2=0
	gg2=[[]]
	
	def incr(g, ctr2=ctr2, gg2=gg2):
		for i in gg2[-1]:
			print 'i', i
			if (g == i):
				print '== i=',i,'in ',gg2[-1],'g= ', g, ctr2
				ctr2 = ctr2+1 
				gg2.append([]); gg2[-1].append(g);
				#print gg[-5:], ctr
				return ctr2
		else:			
			print 'NOT ','g= ', g, ctr2
			gg2[-1].append(g)
			return ctr2
	return incr

class Grapes(object):
	ctr2 = 0
	gg2=[[]]

	def sum(self):
		Grapes.ctr2 +=1

	def appin(self,g):
		Grapes.ctr2 += 1
		Grapes.gg2.append([])
		Grapes.gg2[-1].append(g)

	def appout(self,g):
		Grapes.gg2[-1].append(g)

	def uniquegroups(self,g):
		'''g is row value from itemid'''
		for i in self.gg2[-1]:
			if (g == i):
				self.appin(g)
				return Grapes.ctr2
		else:			
			self.appout(g)
			return Grapes.ctr2


def sort():
	'''merge sort time cols'''
	'''group by range cut'''

	'''set indices by patientid, unstacked itemid to columns   '''





pt['realtime'] = pt.realtime.map(lambda x: pd.datetools.parse(x).time())
#parse(pt.realtime.ix[0])
pt.groupby('realtime')
#pt['group'] = pt['itemid'].map(lambda x: uniquegroups(x) )
ugr = Grapes()
print 'UGR ', ugr
print ugr.uniquegroups(2)
pt['group'] = pt.apply(lambda row: ugr.uniquegroups(row['itemid']), axis=1 )
#pt['realtime','group','itemid'][:5]
pt.head()
#pt['group'] = pt.apply(lambda row: uniquegroups(row['itemid']), axis=1 )



"""
df = DataFrame({'time1':  , 'time2':  ,'item':    } ) 


global ctr
global gg
#gg[0].append([])
ctr = 0; 
gg=[]

def uniquegroups(g):
    global gg
    global ctr
    gg.append([])
    #print gg[-1]
    for i in gg[-1]: 
        if g == i:
            ctr=ctr+1 
            gg.append([])
            gg[-1].append(g)
            break
    else:
        gg[-1].append(g)
    
    return ctr
    
pt['realtime'] = pt.realtime.map(lambda x: pd.datetools.parse(x).time())
#pt.groupby('realtime')/
#parse(pt.realtime.ix[0])
pt.groupby('realtime')
#pt['group'] = pt['itemid'].map(lambda x: uniquegroups(x) )
pt['group'] = pt.apply(lambda row: uniquegroups(row['itemid']), axis=1 )
pt[['realtime','group','itemid']].head()


def uniquegroupsA(g):
	#g is row value from itemid
	#print('row value is',g)
	global gg
	for i in gg[-1]:
		if (g == i):
			print '== i=',i,'in ',gg[-1],'g= ', g, ctr
	#		global ctr
	#		ctr = ctr+1 
			increase()
			gg.append([]); gg[-1].append(g);
			#print gg[-5:], ctr
			return ctr
	else:			
		print 'NOT  i= ',i,'g= ', g, ctr
		gg[-1].append(g)
		return ctr
    
#def uniquegroups2(g):
#	#g is row value from itemid
#	#print('row value is',g)
#	global gg
#	global ctr
#	for i in gg[-1]:
#		if (g != i):
#			print 'NOT i=',i,'in ',gg[-1],'g= ', g, ctr
#			gg[-1].append(g)
#			return ctr
#	else:			
#		ctr = ctr+1 
#		gg.append([]); gg[-1].append(g);
#		print '==  g= ', g,gg[-1], ctr
#		return ctr
#
def agroup(g):
	# - key=itemid : vals=groupid
	ctr = 0; gg=[]
	for i in gg:
		if g==g[i]:
			ctr+=1; g[:]=[]
			return ctr
	else:
		gg.append(g)
		return ctr

def uniquegroups(g):
    # - key=int : val=[a,b,c]
    d = defaultdict(list)
    key=0; d[0]=[];
    def newfunc():
        for k in d:
            if g not in d.items:
                d[key].append(g)
                return key
        else:
            key+=1
            d[key].append([])
            d[key].append(g)
            
        return key
    return newfunc()
			


def uniquegroups(g):
	# - key=int : val=[a,b,c]
    d = defaultdict(list)
	key=0
	return def newfunc():
		if g not in d[key]:
			d[key].append(g)
		else:	
			key+=1 
			d[key].append(g) 
			
    	return key


pt['group'] = pt['itemid'].map(uniquegroups)



def uniquegroups1(x):
    grp = 1
    dct = []
    def unq():
        if x not in dct:
            dct.append(x)
            return grp
        else: 
            dct[:] = []
            grp += 1
            return grp
    return unq
"""
