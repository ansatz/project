'''
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

1. ok get that in files for 

WHAT IS OUTPUT? >> push data to csv format and load into R

> alert vs time
> summary graph
>


2. get 1000 rows returned
have to list all the patients except last one, rerun
mv from ~/Downloads/export... ~/project/data/expired/
output list of patients to not include total
'''


from subprocess import call
import os,csv


# exportqueryresults.csv   --> project/data/expired/male  --> project/data/expired/female
try:
	cmd_mv = 'find /home/solver/Downloads/ -type f -name 'exportqueryresult*' -exec mv {} --target-directory=/home/solver/project/data/expired/female/' +  nofiles+1 +'.csv \;'
	
	call(cmd_mv, shell=True)

except:
	print "copy dwnld file error"

# cwd -> #files
cwd = os.getcwd()
files=next(os.walk(cwd))[2] #relative path
css = [ c for c in files if('.csv' in c) ]
if ! len(css): 
	nofiles=0
else:
	nofiles = len(css)

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
		pts[i]=pts[i][:-1]

print 'ids :', css
print " "
for pp in pts:
	for p in pp: 
		print p, ' ,'






