'''
-- sql
select ce.subject_id, 
       pt.sex, pt.dob, pt.dod, pt.hospital_expire_flg, ce.itemid, 
       ci.description, charttime, realtime, 
       value1num,value1uom, value2num,value2uom
from chartevents ce,
     d_chartitems ci, 
     d_patients pt
where hospital_expire_flg='Y' and pt.sex='F'
  and ce.subject_id not in (1381)
  and ce.itemid in (455,1148,618,115,211,3581)
  and ce.itemid=ci.itemid 
  and pt.subject_id=ce.subject_id


> WHAT IS OUTPUT? >> push data to csv format and load into R
>
> alert vs time
> summary graph
>
'''
from subprocess import call
import os,csv

## cwd -> #files
cwd = os.getcwd()
files=next(os.walk(cwd))[2] #relative path
css1 = [ c for c in files if('.csv' in c) ]
if len(css1)==0: 
	nofiles=0; print 'nofiles: ', nofiles
else:
	nofiles = len(css1)

# exportqueryresults.csv   --> project/data/expired/male  --> project/data/expired/female
cmd_mv = 'find /home/solver/Downloads/ -type f -name exportqueryresults.csv -exec mv {} %d.csv \;' % (nofiles+1)
try:
	call(cmd_mv, shell=True)
except:
	print "copy dwnld file error"

#update files after mv new file into dir/
cwd = os.getcwd()
files=next(os.walk(cwd))[2] #relative path
css = [ c for c in files if('.csv' in c) ]
if len(css)==0: 
	nofiles=0; print 'nofiles: ', nofiles
else:
	nofiles = len(css)

# print patients to remove
pts=[]
for i,cs in enumerate(css):
	pts.append([])
	with open( cs, 'rb') as csvfile:
		rd = csv.reader(csvfile)

		for j,row in enumerate(rd):
#			print 'j ', j, row[1]
			if j == 0:
				pts[i].append( row[1] )
			elif j>0 and row[1] != pts[i][-1] :
				pts[-1].append(row[1])
		#print 'ids :', cs, pts[-1]

pts[i]=pts[-1][:-1]

print 'ids :', css
print " "
for pp in pts:
	for p in pp: 
		print p, ' ,'






