import Image
from itertools import *
import numpy as np
from collections import Counter

#1D
#def check(in):

def entCnt( *args ):
        c=Counter(); wht = np.copy(*args); arln = 1.0/4150
        #dict key:values=count-frequency
        for e in wht:
                c[e]+=1
        #ascending sort by key
        srt=np.unique( sorted(c) )
        #print 'len(wht) ', len(wht)
        #print 'entro srt ' , srt[0:10]
        #count number of val in interval=arln
        frc=[] ; tvr=0
        for i in srt:
                if i < arln:
                        tvr += c[i]
                        #print 'pronto ', i, ' < ',arln, ' ', c[i], ' ',tvr
                else:
                        frc.append(tvr)
                        tvr=0
                        arln+=arln
                        #print 'subji ', arln
        #print 'frc ', frc[0:10]
        etr = np.sum([ ((p * np.log2(p)) ,0.0) for p in frc if p>0.0])
        #print 'entropy ', etr
        return etr

def entroNumpy( *args ):
        wht = np.copy( *args )
        unq = np.unique( wht )
        #print("unq", unq)
        lent = unq.shape[0]

        weightLen = wht.shape[0] * 1.0
        probal = np.zeros(( unq.shape ), dtype='float')

        itr = 0

#        for i in xrange(0,1,1/1000):



        for u in unq:
                #print('u',u)
                indx = np.where( wht == u )  #count for x
                shp = len( wht[indx] )
                #print( wht[indx], shp , weightLen)
                probal[itr] = ( shp / weightLen )  #normalized count
                itr = itr + 1

        #print("probal", probal)
        sze = probal.shape[0]
        entropy1 = np.zeros( sze )

        entropy1 = -np.sum([ ((p * np.log2(p)) ,0.0) for p in probal ])
        return entropy1
#2D
def f_entrpy( *args ):
        wht = np.copy( *args )
        rows = wht.shape[0]
        clm = wht.shape[1]
        entropy = np.zeros(clm,dtype='float')
        jj=0
        for col in wht.T:
                #print("col", col)
                entropy[jj]= entroNumpy(col)
                #print( jj, entropy[jj] )
                jj+=1
        return entropy

#############################################
def image_entropy(img):
   """calculate the entropy of an image"""
   histogram = img.histogram()
   histogram_length = sum(histogram)
   samples_probability = [float(h) / histogram_length for h in histogram]
   return -sum([p * math.log(p, 2) for p in samples_probability if p != 0])

def image2array(im):
   newArr = np.fromstring(im.tostring(), np.uint8)
   newArr = np.reshape(newArr, im.size)
   return newArr

def PIL2array(img):
   return np.array(img.getdata(),
                   np.uint8).reshape(img.size[1], img.size[0], 3)

def array2PIL(arr, size):
   mode = 'RGBA'
   arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
   if len(arr[0]) == 3:
      arr = numpy.c_[arr, 255 * numpy.ones((len(arr), 1), numpy.uint8)]
   return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)

#def name():
#   """
#   """
#from sklearn.utils import assert_all_finite
#from sklearn.utils import safe_asarray

def entropy(*args):
   xy = zip(*args)
   #print("xy", xy)
#xy = np.core.records.fromarrays( [*args] )
   # probs
   proba = [ float(xy.count(c)) / len(xy) for c in dict.fromkeys(list(xy)) ]
   safe_asarray(xy)
   #very pythonic list comprehension
   # the follwoing line is just a list comprehnsion with x =
   # x[numpy.isfinite(x)] having ability to filter crap out
   # x = x[numpy.logical_not(numpy.isnan(x))]
   #list comprehension with if-condition
   entropy = -np.sum([ ((p * np.log2(p)) , 0 )    [ math.isnan(p * np.log2(p)) or math.isinf(p * np.log2(p)) ] for p in proba ])
   #assert_all_finite(entropy)
   return entropy



#ent = np.array([])
#ent = [] #[wghtAll.shape[1]]
##for i in xrange( wghtAll.shape[1] ):

#for i in xrange(WTS): #.shape[1]):  #over col
    #entropy over col is scalar, so e = is OK! ie not an array
#for i in WTS.shape[1]:

#print( "type(WTS)" , type(WTS) )
#print( "WTS.shape" , WTS.shape)
#print WTS

#print "entropy loop...check your input size mimic=4150 | th=2150 "
#X2 = np.zeros((4150,6),'float64') #mimic 4150,6 th2150,6
#fit2(clf,patData,patTarg)
#print("WTS shape", WTS.shape)
#X2 = WTS.copy()
#print("X2", X2.shape)
##ent = np.zeros((4150,6),'float64' )   #mimic 4150,6  th2150,6
#ent = np.zeros((X2.shape[1]),'float64')
#print("ent shape/for", ent.shape )
##for i in xrange(X2.shape[0]):
##        ent_temp = np.zeros((X2.shape[1]),'float64')
##        print i
##        if i == 0: #X2.shape[0]-1:
##               print("i=",i)
##               print("ent_temp", ent_temp)
##        for j in xrange(X2.shape[1]): #6 datapoints
##                #e = entropy(WTS[:, i])   #print("++++"); print e; print("----")
##                e = entropy( X2[:,j] ) #BC n_estimators #100 #ent_temp = np.hstack((ent,/e)) #np.append( ent_temp, e )
##                print("e",j, e)
##                ent_temp[j] = e
##        ent[i]= ent_temp
##        print( "ent_temp", ent_temp[j])
##        print("ent[i]", ent[i] )
##        break
###ent is vector of size X2.shape (4150=mimic 2150=th)
#swunk = np.zeros( (X2.shape[1]) )
#for i in xrange( X2.shape[1] ):
#        ent[i]= entropy( X2[:,i] )
#        #ent[i]= entroNumpy( X2[:,i] )
#        swunk[i]= ent[i]
#
#print("swunk", np.unique(swunk), len(swunk) )
#entUnique = np.unique(ent)
#noEntUnique = len(entUnique)
#print("entropy unique, number", entUnique, noEntUnique)
#
#print "end for"
##ent = entropy( patData[:,j] ) for
##binary = cPickle.dumps(a,) # cPickle.loads
##np.tostring(ent) #np.fromstring(ent)
#lent= len(ent)
#print lent
#print ent
#print ("**entropy loop end")



#entr = np.array( [2,4,7,7,8,9] )
#entrr = np.array(((2,4,7,7,54,8,9),(2,4,7,37,7,8,9),(1,2,2,3,4,5,6),(0,23,23,52,55,58,63)))
#print("entr", entr)
#e = entroNumpy(entr)
#ea = entroNumpy(entrr)
#print( e, ea )
#entr2 = np.array(((2.34,4.56,6.77,88.),(1.1,3.3,50,70)))
#print entr2[:,0]
#e2= f_entrpy(entrr)
#print e2
