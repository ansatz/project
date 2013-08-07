#################IMPORTS LIBRARIES#########################################
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#import pdb
import Image
import scipy
#import fastKDE as Kernel
import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_equal

from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import BoostedClassifier
#from sklearn.ensemble import *
from sklearn.base import ClassifierMixin
from sklearn import datasets
import math

#from pyentropy import DiscreteSyste
from scipy import stats
from collections import defaultdict
WTS = np.array([]); INC = np.array([]);
###################    FUNCTIONS   ############################################

def predict_proba(BC, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the weighted mean predicted class probabilities
        of the trees in the forest.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples]
            The class probabilities of the input samples. Classes are
            ordered by arithmetical order.
        """
        X = np.atleast_2d(X)
        p = np.zeros((X.shape[0], self.n_classes_), dtype=np.float64)
        norm = 0.
        for alpha, estimator in zip(self.boost_weights, self.estimators_):
            norm += alpha
            if BC.n_classes_ == estimator.n_classes_:
                p += alpha * estimator.predict_proba(X)
            else:
                proba = alpha * estimator.predict_proba(X)
                for j, c in enumerate(estimator.classes_):
                    p[:, c] += proba[:, j]
        if norm > 0:
            p /= norm
        return p


def fit2(BC, X, y, sample_weight=None, **kwargs):
         X = np.atleast_2d(X)
         y = np.atleast_1d(y)
#print("boost says hello")
         if isinstance(BC.base_estimator, ClassifierMixin):
            BC.classes_ = np.unique(y)
            BC.n_classes_ = len(BC.classes_)
            y = np.searchsorted(BC.classes_, y)

         if not sample_weight:
            # initialize weights to 1/N
            sample_weight = np.ones(X.shape[0], dtype=np.float64)\
                / X.shape[0]
         else:
            sample_weight = np.copy(sample_weight)

            # boost the estimator
            # Currently only AdaBoost is implemented, using the SAMME modification
            # for multi-class problems

         #WTS np.ones(X.shape[0])
         RV = []
         global WTS
         global INC
         WTS = np.zeros((BC.n_estimators, X.shape[0]),'f')
         INC = np.zeros((BC.n_estimators, X.shape[0]),'f')
         #WTS = np.copy(sample_weight)
         #print("row,col,WTS", BC.n_estimators, X.shape[0], WTS.shape)
         for i in xrange(BC.n_estimators):
            estimator = BC._make_estimator()
            estimator.fit(X, y, sample_weight, **kwargs)
            #WTS[i:] = sample_weight
            #print(i)
            #print( WTS[i:] )
            # TODO request that classifiers return classification
            # of training sets when fitting
            # which would make the following line unnecessary
            T = estimator.predict(X)
            #print("Ti",i, T)
            # instances incorrectly classified
            if BC.two_class_cont:
               incorrect = (((T - BC.two_class_thresh) * \
                                (y - BC.two_class_thresh)) < 0).astype(np.int32)
            else:
               incorrect = (T != y).astype(np.int32)
            #INC = np.vstack([INC, incorrect]) #adding row
            #print("y", y)
            #print("INCi", i, incorrect)
            # error fraction
            err = np.sum(sample_weight * incorrect) / np.sum(sample_weight)
            #print("err", err)
            # sanity check
            if err == 0:
               BC.boost_weights.append(1.)
               #print("ERROR", err, i )
               #err += 0.05
               break
            elif err == 0.5:
               if i == 0:
                  BC.boost_weights.append(1.)
                  #err -= 0.5
               #print("ERROR", err, i)
               break
            # boost weight using multi-class SAMME alg
            alpha = BC.beta * (math.log((1 - err) / err) + \
                                        math.log(BC.n_classes_ - 1))
            BC.boost_weights.append(alpha)
            #bbc = np.asarray(BC.boost_weights)
            WTS = np.vstack((WTS,sample_weight))
            #print("a", alpha )
            if i < BC.n_estimators - 1:
               #print("i", i)
               correct = incorrect ^ 1
               sample_weight *= np.exp(alpha * (incorrect - correct))
               #print( "a", alpha )
               #WTS = np.vstack([WTS, sample_weight])
               #print( "wts", WTS[i: ]
            #print("i",i)
            RV.append(WTS)
            RV.append(INC)
         return RV

#search hard/easy
#def HEsY_alignmentODDratio ():
#{
       # wts = WTS.copy()
        #inc = INC.copy()
        #GLOBS{sys,dia,hr1,ox,wght,hr2}:
        #{cilantro}: classifier_entropy




#}






# toy sample
#X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
#print X[0][1] #-1
#print X[5][0] # 2
#y = [-1, -1, -1, 1, 1, 1]
#T = [[-1, -1], [2, 2], [3, 2]]
#true_result = [-1, 1, 1]

# also load the iris dataset
# and randomly permute it
#iris = datasets.load_iris()
#np.random.seed([1])
#perm1 = np.random.permutation(iris.target.size)
#perm2 = np.random.permutation(iris.target.size)
#iris.data = iris.data[perm1]
#iris.target = iris.target[perm2]

#print("hello")###########

# also load the boston dataset
# and randomly permute it
#boston = datasets.load_boston()
#print boston
#perm = np.random.permutation(boston.target.size)
#boston.data = boston.data[perm]
#boston.target = boston.target[perm]


# Adaboost Classification

clf = BoostedClassifier(n_estimators=100)


# FILE INPUTS ###z
#WTSS = fit2( clf, X, y )
#print WTSS
   #print("iris.data", iris.data)
   #print("iris.target",iris.target)
   #RVV = fit2( clf, iris.data, iris.target )
#WTSS = RVV[0]
#INCC = RVV[1]
#WTSS = fit2( clf, boston.data, boston.target )
#print( "INC", INCC[:,99:100] )
#print( "weights", WTSS[:,99:100] )
#print( "INC", INCC[:,39:40] )
#print( "weights", WTSS[:,39:40] )
#print( "wts", WTSS[7:8,:])

import pylab as pl
import scipy.special as sps
#pl.hist(WTSS[:,149:150], 100, label= 'histogram' )
  #pl.hist(100,WTSS[:,149:150],label='hist')
  #pl.show()

  #a=2.
  #s = WTSS[:,49:50].T
  #s.reverse()
  #print("s",s)
  #count, bins, ignored = pl.hist( s[s<50], 50, normed=True)
  #x = np.arange(1., 50.)
  #y = x**(-a)/sps.zetac(a)
  #pl.plot(x, , linewidth=2, color='r')
  ##pl.show()
  #print( "helloinput")
  ######################3
  #import ~/Desktop/input
  #import csv
  #raw = np.asarray(list( csv.reader( open( r'./input2.csv') ) ))
  #float(raw)
  #print raw.shape
  #print raw
  #raw2 = raw[1:-1 , :]
  #print raw.shape
  #rdata = raw[:, 0:-2]
  #rtarget = raw[ :, 7]
  #print rdata
  #print rtarget
  #pat =  csv.reader(open('input2.csv','r'))
  #for row in pat:
  #   for i in row:
  #      float(i)
  #   pat.append(rowfrom pprint import pprint
import csv
def parseNum(x):
    xx = x.replace(",", "")
    #if not xx.replace(".","").isdigit(): return x
    return "." in xx and float(xx) or int(xx)


###################
x = [map(parseNum, line) for line in csv.reader(open("/home/solver/Desktop/data/raw-labeled-th/all.csv"))]
#x = [map(parseNum, line) for line in csv.reader(open("all-th.csv"))] #smooth-th
#x = [map(parseNum, line) for line in csv.reader(open("/home/solver/Desktop/data/smooth-mii/all.csv"))]
#x = [map(parseNum, line) for line in csv.reader(open("/home/solver/Desktop/data/raw-labled-mimic/all.csv"))]
##(open("\.\./all\.csv"))]
print("250: x.shape", len(x))
print("2150 is a lucky number")

bst = np.asarray(x)
bstRows = bst.shape[0]
bstCol = bst.shape[1]; print bstCol
patData = bst[:, 0:bstCol-1].copy()  #everything but last column(labels)
patTarg = bst[:, bstCol-1].copy()
print("l262, patdat pattar", patData.shape, patTarg.shape)
#micin = [map(parseNum, incline) for incline in csv.reader(open("../all2.csv"))]
###flag for cross validate, change title of graphs

flag = True
if flag == True:
        ##########CROSS-VALIDATE^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #import numpy as np
        from scipy import interp
        import pylab as pl

        from sklearn import svm, datasets
        from sklearn.metrics import roc_curve, auc, precision_score
        from sklearn.cross_validation import StratifiedKFold

        ###############################################################################
        # Data IO{} mv{}2 GroupsPLot Together
        # 1.patData,patTarg,
        # 2.mimicData mimicTarg
        # import some data to play with
        #iris = datasets.load_iris()
        X = patData.copy()  #iris.data
        y = patTarg.copy()  #iris.target
        X, y = X[y != 2], y[y != 2]
        n_samples, n_features = X.shape
        print("l284: X.shape", X.shape)
        # Add noisy features T!
        X = np.c_[X, np.random.randn(n_samples, 200 * n_features)]

        ###############################################################################
        # Classification and ROC analysis

        # Run classifier with crossvalidation and plot ROC curves

        ####cv = fit2(clf,patData,patTarg)
        cv = StratifiedKFold(y, k=6)
        #classifier = svm.SVC(kernel='linear', probability=True)
        classifier = BoostedClassifier(n_estimators=100)
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        all_tpr = []

        print "begin train-cv.."
        #train
        for i, (train, test) in enumerate(cv):
                probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
                         #fit2(clf, X[train], y[train]) !#probas_ = classifier.predict_proba(X[test]) !
                # Compute ROC curve and area the curve
                fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
                mean_tpr += interp(mean_fpr, fpr, tpr)
                mean_tpr[0] = 0.0
                roc_auc = auc(fpr, tpr)
                pl.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
                #pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
        print "end train-cv"
        mean_tpr /= len(cv)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        pl.plot(mean_fpr, mean_tpr, 'k--',
                label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

        pl.xlim([-0.05, 1.05])
        pl.ylim([-0.05, 1.05])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title('Reciever Operator Curves, CV, and AUC for Telehealth Patient Group ')
        #pl.title('Reciever Operator Curves, CV, and AUC for MIMIC-II Patient Group')##########fck
        pl.legend(loc="lower right")
        pl.show()

       # } ##end of cross validation



#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#print patTarg.np.type
#print patTarg
#rvAll = [ [i] for i in ( fit2(clf, patData, patTarg) )]
#rvAll = np.fromfunction(fit2(clf,patData,patTarg))
#fit2(clf,patData,patTarg)

#wghtAll=rvAll[0]
#print("WTS")
#print WTS.shape
#pl.hist(wghtAll[:,5:6], 100, label= 'histogram' )
#pl.hist(wghtAll[:,5:6], 50)
#pl.show()

##################################################################################
#entropy-histogram
  #ROWS ARE BOOSTED WEIGHT ITERATIONS
  #ENT-FX = [TPL_0,TPL_1,TPL_2....TPL_N]
  #INPUT: M(BOOST-ITR) X N(TUPLES)
  #OUTPUT: ARRAY (1 X N)

  #def ent_his(WTS binsL):

  #for i in WTS.shape[1]:  #col
  #	len = 1/binsL
  #	for j in WTS.shape[0]:  #rows
  #		data = WTS[:,j]     #[0, 0.5, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5, 3]
  #		hist, edges = np.histogram(data, bins=binsL)
  #
  #		getbin = 2  #0, 1, or 2
  #
  #	for d in data:
  #    	val = np.searchsorted(edges, d, side='right')-1
  #    	if val == getbin or val == len(edges)-1:
  #       		print 'found:', d
  #		#end if
  #	#end for
   #
##import sys
##sys.exit()
##pdb.setTrace()

 ###########################################################
import numpy
import Image
from itertools import *

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
from sklearn.utils import assert_all_finite
from sklearn.utils import safe_asarray
def entropy(*args):
   xy = zip(*args)
   # probs
   proba = [ float(xy.count(c)) / len(xy) for c in dict.fromkeys(list(xy)) ]
   safe_asarray(xy)
   #very pythonic list comprehension
   # the follwoing line is just a list comprehnsion with x =
   # x[numpy.isfinite(x)] having ability to filter crap out
   # x = x[numpy.logical_not(numpy.isnan(x))]
   entropy = -np.sum([ ((p * np.log2(p)) , 0 )    [ math.isnan(p * np.log2(p)) or math.isinf(p * np.log2(p)) ] for p in proba ])
   assert_all_finite(entropy)
   return entropy

ent = np.array([])
#ent = [] #[wghtAll.shape[1]]
##for i in xrange( wghtAll.shape[1] ):

#for i in xrange(WTS): #.shape[1]):  #over col
    #entropy over col is scalar, so e = is OK! ie not an array
#for i in WTS.shape[1]:

#print( "type(WTS)" , type(WTS) )
#print( "WTS.shape" , WTS.shape)
#print WTS
X2 = patData.copy()
print("X2", X2.shape)
for i in xrange(X2.shape[1]):
        #e = entropy(WTS[:, i])   #print("++++"); print e; print("----")
        e = entropy( patData[:,i] )
        ent = np.hstack((ent,e))
#efor

print("437")
lent= len(ent)
print lent
print ent
#######################################################
#import pickle
#with open('rvAll.pickle', 'wb') as f:
 #  pickle.dump(entry, f)
import pandas
import statsmodels.api as sm

 #print rvAll.head(10)
 #a = rvAll[0]
 #a.describe()

####KDE-DENSITY ESTIMATE/ SMOOTHING OF ENTROPY HISTOGRAM###
import matplotlib.pyplot as plt
##plt.imshow(wghtAll) #Needs to be in row,col order
  #plt.savefig(WI)
  #print("entropy-normal")
  ##i= Image.fromarray(wghtAll)
  #img2 = Image.open(WI)
  #print image_entropy(img2)
  #
  ##sze = wghtAll.hape[0] * wghtAll.shape[1]
  #sze = sum(len(xx) - 1 for xx in x)
  #iii = array2PIL( wghtAll, len(wghtAll)  )
  #imggg = Image.open(iii)
  #print image_entropy(imggg)
  #
  ##img = Image.open(wghtAll[:,100:101])
  #entropyArr = []
  #for i in xrange( wghtAll.shape[0] ) :  #rows
  #   if (i == wghtAll.shape[0] ):
  #       break
  #   #print wghtAll[:, i:i+1].shape
  #   img0 = wghtAll[:,i:i+1]
  #   img = Image.fromarray(img0) #img is image
  #   entropyArr.append( image_entropy(img) )
  #
  ##print img0.shape
  #print("entropy:")
  #from numpy import array
  #entropyArr1 = array(entropyArr)
  #print entropyArr1
  #print entropyArr1.shape
  ##print wghtAll[:,100:101].shape
  #
  ##image2array(entropyArr)
  ##PIL2array(entropyArr)
  ##print entropyArr.shape
from pylab import plot,show,hist,close
from matplotlib.pyplot import imshow
from scipy import stats
  #def measure(n):
  #    "Measurement model, return two coupled measurements."
  #    m1 = np.random.normal(size=n)
  #    m2 = np.random.normal(scale=0.5, size=n)
  #    return m1+m2, m1-m2

  #m1, m2 = measure(2000)
  #xmin = m1.min()
  #xmax = m1.max()
  #ymin = m2.min()
  #ymax = m2.max()


  #data = entropyArr1
  #data = ent
  #print data.shape
  #enta = np.array([])
  #print ent[975]
  #enta = map(float, ent)
  #xmin,ymin = np.min(ent) #np.min(xargs=ent)
  #xmax,ymax = np.max(ent)
  #xmin, ymin = 1.0
  #xmax, ymax = 2.0
  #ental = long(ent)
  #print("happy2cu")
  #print type(ent)

  # Perform a kernel density estimate on the data:
  #X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
  #positions = np.vstack([X.ravel(), Y.ravel()])
  #values = np.vstack([m1, m2])
kernel = stats.gaussian_kde(ent)
#plt.figure()
################################################change her for different versions
#plt.title("Kernel Density Estimate with Hist.over ALL boosted-weight Entropy")
#plt.title("KDE: tele r cv")
plt.title("KDE: tele s cv")
#plt.title("KDE: tele r -cv")
#plt.title("KDE: tele: s -cv")




#kpdf = kernel.evaluate(ent)
kdeX = np.linspace(0,5,100)
#xlog = np.logspace(-5,5,100)


print("****   kde   ****  ")
plot(kdeX,kernel(kdeX),'r',label='kde') # distribution function
#plt.plot(x, stats.norm.pdf(x), color="g", label='generator')
#plt.plot(x, kpdf, color="grey", label="pdf")
hist(ent,normed=1,alpha=.3) # histogram over all points

#plt.savefig('kde.png')

plt.show()
import sys
sys.exit()

#close()

#not-possible with 6dimensions
#------2D-SCATTER-WITH KDE PLOT+HEATMAP--------
#how to unit-test vectorize ?  repl is the answer!!!!
#relation-test: equality  how do test if two fncs are equal,,, with if condition
#termination-test:
#loop-test:

#import pylab as pl
#from numpy import *
import matplotlib.pyplot as plt
#sze = WTS.shape[1]
sze = X2.shape[1]
sctX = np.linspace(0, 20, sze)
#def f(x):
#  return x
# f(x) will in general not work
####f_vec = np.vectorize(x) #will vectorize any string object!!!
#return x**x*4 vectorizes as well
#sctY = f(sctX)
#scatterplot( sctX, sctY, s=sze, cmap=cm.hsv)

#f_vec =


#-----------------------------------------------
###########SPLOM>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<
import matplotlib.pyplot as plt

#detects hard/easy points based on threshold
#with entropy calc. for each sequential boosted classifier
print ("chk-clear")


def scatterplot(data, data_name ):
    """Makes a scatterplot matrix:
    Inputs:
      data - a list of data [dataX, dataY,dataZ,...];
             all elements must have same length

      data_name - a list of descriptions of the data;
                  len(data) should be equal to len(data_name)

    Output:
      fig - matplotlib.figure.Figure Object

    """
    N = len(data)
    fig = plt.figure()
    #####################################
    ##############titles################
    plt.title("xx:")
    #plt.title("mii: scatter plot matrix, kde-entropy hist")
    #plt.title("tele: scatter plot matrix, kde-entropy hist"")
    plt.title("xx scatter plot matrix, kde-entrpy hist")
    #plt.title("mii-cv scatter plot matrix, kde-entrpy hist")

    print "splom-chk"
    for i in range(N):
        for j in range(N):
            ax = fig.add_subplot(N,N,i*N+j+1)

            if j == 0: ax.set_ylabel(data_name[i],size='5')
            if i == 0: ax.set_title(data_name[j],size='8')
            if i == j:
                    ax.hist(data[i], 10, lw=1)
                #kernel = stats.gaussian_kde(ent)
                   # ax.plot(data[i],kernel(data[i]),'r',label='Kernel.Density.Estimate') # distribution function
            else:
                   # color = [str(item/255.) for item in y]
                    ax.scatter(data[j], data[i]) #, s=500, c=color)

    return fig

# Example
#if __name__ == "__main__":
#    import numpy as np
#    import numpy.random as npr
#
#    X = npr.randn(100)
#    Y = 1.2 * X + npr.normal(0.0, 0.1, 100)
#    Z = - Y ** 2 + X + 0.05 * npr.random(100)
#    W = X + Y - Z + npr.normal(0.0, 2.0, 100)
#
#    data = [X, Y, Z, W]
#    data_name = ['Data X', 'Data Y', 'Data Z', 'Data W']
#
#    fig = scatterplot(data, data_name)
#
#    fig.savefig('scatterplot.png', dpi=120)
#    plt.show()
#patData
#patTar
#if __name__ == "__main__":
import numpy as np
import numpy.random as npr
alsarry= patData.copy()

#X = npr.randn(100)
#Y = 1.2 * X + npr.normal(0.0, 0.1, 100)
#Z = - Y ** 2 + X + 0.05 * npr.random(100)
#W = X + Y - Z + npr.normal(0.0, 2.0, 100)

X = alsarry[0]
Y = alsarry[1]
Z = alsarry[2]
W = alsarry[3]
A = alsarry[4]
B = alsarry[5]
data = [X, Y, Z, W, A, B]
data_name= ['SYSI', 'DIAI', 'HR1I', 'OXI', 'HR2I','RESP' ]

fig = scatterplot(data, data_name)
#fig.savefig('scatterplot.png', dpi=120)

plt.show()
##################################################################################
#->>>>>>>>>>>>>>>>>>>>>>>>>>>>>------------------------------
#from pylab import subplot, scatter, hist, grid
#import numpy as np
#
#def splom(data, targets=None):
#  (N, D) = data.shape
#  if targets == None:
#    targets = np.zeros(N)
#  for i in range(D):
#    for j in range(D):
#      subplot(D, D, i*D + j + 1)
#      if i == j:
#        hist(data[:,i], bins=20)
#      else:
#        scatter(data[:,j], data[:,i], c=targets)
#        grid()
#
#p2=patData.copy()
#fig2=splom(p2, 'None')
#fig2.savefig('splom.png',dpi=90)
#plt.show()

#import sys
#sys.exit()
#pdb.setTrace()


#####KERNELDENSITY-SMOOTHER#########################
#import numpy as np
#x, y = np.mgrid[-10:10:100j, -10:10:100j]
#r = np.sqrt(x**2 + y**2)
#z = np.sin(r)/r
#r = kernel( kernel(kdeX) )
####
#def e2(entropy, x , y):
#        for i in xrange(WTS.shape[1]):  #over col
#                e = entropy(WTS[:, i])
#                #ent.append(e)
#:                ent = np.hstack((ent,e)


#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
# Twice as tall as it is wide.
#from enthought.mayavi import mlab
#from mayavi import mlab
#from enthought.tvtk.tools import mlab
#fig = plt.figure(figsize=plt.figaspect(2.))
#fig.suptitle'Kernel-Density-Estimate: over all points')
#ax = fig.add_subplot(2, 1, 1, projection='3d')
#ax.grid(True)
#Xx = np.arange(-5, 5, 0.25)
#xlen = len(X)
#Yy = np.arange(-5, 5, 0.25)
#ylen = len(Y)
#Xx, Yy = np.meshgrid(Xx, Yy)
#R = np.sqrt(Xx**2 + Yy**2)
#print R.entropy()
#Z = np.sin(R)
##surf = ax.plot_surface(x, y, z, rstride=1, cstride=1,
##        linewidth=0, antialiased=False)
#grph = surf( ent )
#ax.set_zlim3d(-1, 1)
#show()

from scipy import *

#[x,y]=mgrid[-5:5:0.1,-5:5:0.1]
 #r=sqrt(x**2+y**2)
 #z=sin(3*r)/(r)

from enthought.tvtk.tools import mlab
import numpy
from enthought.mayavi.mlab import *
# Open a viewer without the object browser:
#f=mlab.figure(browser=False)
#ent2d = np.hstack((ent,ent))
##s=mlab.Surf(x,y,z,z)
#s = surf(ent2d)
#f.add(s)
#s.scalar_bar.title='smooth threshold'
#s.show_scalar_bar=True
## LUT means "Look-Up Table", it give the mapping between scalar value and color
#s.lut_type='blue-red'
## The current figure has two objects, the outline object originaly present,
## and the surf object that we added.
#f.objects[0].axis.z_label='value'
#t=mlab.Title()
#t.text='Sampling function'
#f.add(t)
## Edit the title properties with the GUI:
#t.edit_traits()

from numpy import ma
x,y = np.mgrid[-3:3:100j, -3:3:100j]
#xalist = ma.array([], shrink=False)
l = len(WTS)
#print l
#print WTS.shape
[xalist,y] = mgrid[-2150.:2150.:0.50,-2150.:2150.:0.51]
#[xalist,y]=mgrid[-2150.:2150.:0.50,-2150.:2150.:0.51]
#ma.masked_inside(xalist, 0.5 , -0.5)
#xalist = [float(num[0.]) for x in xalist for num in x ]
#numb_list = ma.array([], shrink=False)
#ma.masked_inside(numb_list, 0.3, -0.3)
#for item in xalist:
#        for numb in item:
#                #numb_list.append(float([numb[0]]) )
#                numb_list = ma.getmaskarray(float([numb_list[0] ] ) )
#a = 0.7
#a0 = 0.53
#r1=np.sqrt((x-a)**2 + y**2)
#r2= np.sqrt((x+a)**2 + y**2)
#val = np.exp(-r1/a0) + np.exp(-r2/a0)
#def f_vectorized(r):
#        x1 = np.exp(-r1/a0)
#valVec = np.vectorize(V)
#from enthought.mayavi import mlab
#mlab.surf(values, warp_scale='auto')z
#int([x])
####ent3d = [entropy(WTS[:,i]) for i in xalist  ]
#mlab.surf(ent3d, warp_scale='auto')
#mlab.outline()
#mlab.axes()
#mlab.show()


####

#z = kernel(r)/r
#from enthought.tvtk.tools import mlab
#from enthought.mayavi import mlab
#surf = ax.plot_surface(x, y, z, rstride=1, cstride=1,
#        linewidth=0, antialiased=False)
#ax.set_zlim3d(-1, 1)
#plt.show()
#
##mlab.surf(z, warp_scale='auto')
##mlab.outline()
##mlab.axes()

###########KERNEL-REGRESSION############################################################
#import sys
#sys.exit()
#pdb.setTrace()



#####################################################################
##Kernel method estimates regression f(X) over domain by fitting a different
#model(choose a func) separately. At each x_o query point, using a weighted or kernel
#function, it assigns xi_weight based on distance to xo, using MLE.  The
#kernel is smooth.  Further, p180, autoregressive time series models of order
#k, with denoted lag set z(t-k), can be fit with local LSE.

##------------------------------------
import numpy as np
from sklearn.gaussian_process import GaussianProcess
from matplotlib import pyplot as pl
np.random.seed(1)
from sklearn.utils import as_float_array
from sklearn.utils import safe_asarray
from sklearn.utils import warn_if_not_float

#def f(x):
#    """The function to predict."""
#    return x * np.sin(x)

#X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
#ptDtX = np.atleast_2d(patData).T
# Observations
#y = f(X).ravel()

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
#grid
clmns=20.    #float(clmns)
xKrn = np.zeros(shape=(clmns,1000) )
print("******xKrn", type(xKrn), shape(xKrn) )
xKrn = np.atleast_2d(np.linspace(0., 10., 1000.*clmns))
print("xKrn", type(xKrn), shape(xKrn) )
xKrn.T

#xKrn = xKrn.astype(np.float64, copy=False)
#np.transpose(xKrn)
####xKrn = np.linspace(0.,10.,1000.*clmns)
#np.asarray( xKrn )
#print "dtyp-linspace %f"
#print xKrn.dtype
###xKrn.shape=(1000.,clmns)
#krx = np.linspace(0,10,1000).T
# Instanciate a Gaussian Process model


gp = GaussianProcess(corr='squared_exponential', theta0=1e-2, thetaL=1e-4, thetaU=1e-1, \
                     random_start=100)

# Fit to data using Maximum Likelihood Estimation of the parametersif __name__ == "__main__":
#
#gp.fit(X, y)
#trgExd = [ [i]*6 for i in patTarg  ]
#trgExdf = [i for clm in trgExd for i in clm]
#xxx = [i for clm in patData for i in clm]
#print (patData.ravel().shape[0]) #col
#print len(trgExd)
#print len(trgExdf)
#print patTarg[:][0:12]
#print trgExd[:][0:3]
#print trgExdf[:][0:12]
#trgAry = np.asarray(trgExdf)
#prvl = patData.ravel()
#gp.fit( patData.ravel(), trgAry.T )
#gp.fit( patData.ravel(), patTarg )

#x3 = np.atleast_2d(patData).T
#xxx = np.atleast_2d( patData[90:])     #np.atleast_2d(ent).T
#yyy = np.atleast_2d( patTarg ).T
#yyy3 = patTarg.ravel()  ##2150
#dy = 0.5 + 1.0 * np.random.random(yyy.shape)
#noise = np.random.normal(0, dy)
#yyy += noise
#y3 = predict_proba(clf, patData)
#function generator points
XKrn = np.zeros(shape=(1000,))
XKrn = np.atleast_2d( np.linspace(0.1, 9.9, 1000.) )
np.transpose(XKrn)
#print len(XKrn)
#XKrn = np.atleast_2d(ent).T
#krx[[ ] ]
#b[0] gives col, b[1] gives rows
#a=22
#observations
print ("%%%XKrn shape", XKrn.shape)
yKrn = np.zeros( shape=(1,20000) )
#yKrn = np.array( []) #, dtype=np.float64 )
print("yKrn type", type(yKrn) )
#yKrn = [ entropy(xk)  for xk in xKrn]       #patData.shape([0]) ] #.ravel()
#for i in np.arange(0,1,0.1):
#        print i
for i in xKrn:
        print i
        yKrn[i]= entropy( xKrn[ i:i+clmns ] )
        print xKrn[i]
        i+=clmns
print("##yKrn##", shape(yKrn)  )
print yKrn

np.ravel(yKrn)
safe_asarray(yKrn)   #.ravel()
np.asarray_chkfinite(yKrn)
print("l885: yKrn", type(yKrn) )
#yKrn = yKrn.astype(numpy.float32, copy=False)


#safe_asarray(yKrn).ravel()#print(len(yKrn) )
np.array(yKrn,float);  as_float_array(XKrn);  as_float_array(yKrn)

#yKrn.astype(float)
np.float64(yKrn);  np.asarray( yKrn )
#warn_if_not_float(yKrn)
XKrn = XKrn[np.logical_not(np.isnan(XKrn))]; yKrn = XKrn[np.logical_not(np.isnan(yKrn))]
assert_all_finite(XKrn); assert_all_finite(yKrn)    #X_vec = np.vectorize(XKrn) #y_vec = np.vectorize(yKrn)
XKrn.ravel(), yKrn.ravel()
print("912: XKrn row, yKrn row", XKrn.shape, yKrn.shape )
#new_list =[ (F, T) [boolean test] for x in old_list ]
#chk0 = [ (0.001)  [i == True] for i in yKrn ]


print ("minX", np.min(XKrn) )
print ("y shape", yKrn.shape)
indices =  (y == False).nonzero()       #   np.nonzero(yKrn)
print ("**nonzero yKrn", indices, y[indices])
for i in indices:
        yKrn[i]=1e-10
indx = np.nonzero(XKrn)
for i in indx:
        XKrn[i]=1e-10
indChk1 = np.nonzero((yKrn))
indChk2 = np.nonzero((XKrn))
#print("yrow xrow nonzeros", yKrn.shape, XKrn.shape, indChk1,"\n x zeros \n", indChk2)
#deltas[(deltas<0) | (deltas>100)]=0
yKrn.ravel()
yKrn[(y<0.)]=0.


fitVec = np.vectorize(gp.fit)
fitVec(XKrn, yKrn)
#gp.fit( XKrn, chk0.T )
######gp.fit( XKrn, yKrn )

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, MSE = gp.predict(xKrn, eval_MSE=True)
sigma = np.sqrt(MSE)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
pl.title("Kernel Regression with Std Error=2")
pl.figure()
pl.plot(xKrn, entropy(xKrn), 'r:', label=u'$Entropy:=\sum f_i*log(f_i)$')    #'$f(x) = x\,\sin(x)$')
pl.plot(XKrn, yKrn, 'r.', markersize=10, label=u'Observations')
pl.plot(XKrn, y_pred, 'b-', label=u'Prediction')
pl.fill(np.concatenate([xKrn, xKrn[::-1]]), \
        np.concatenate([y_pred - 1.9600 * sigma,
                       (y_pred + 1.9600 * sigma)[::-1]]), \
        alpha=.5, fc='b', ec='None', label='95% confidence interval')
pl.xlabel('$x$')
pl.ylabel('$f(x)$')
pl.ylim(-10, 20)
pl.legend(loc='upper left')
show()











###########################################################################
#Recursive Feature Selection
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE

# Load the digits dataset
#digits = load_digits()
#X = digits.images.reshape((len(digits.images), -1))
#y = digits.target

# Create the RFE object and rank each pixel
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
#rfe.fit(patData, patTarg)
rfe.fit(X2,y2)
ranking = rfe.ranking_.reshape(patData[0].shape)

# Plot pixel ranking
#defined above at ~500:- import pylab as pl
pl.matshow(ranking)
pl.colorbar()
pl.title(" 'Hard' vs 'Easy' Pts")
pl.show()




  #Z = np.reshape(kernel(positions).T, X.shape)
  #Z = np.reshape(kde(data).T, data.shape)
  # Plot the results:



  #import matplotlib.pyplot as plt
  #fig = plt.figure()
  #ax = fig.add_subplot(111)
  #ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
  #          extent=[xmin, xmax, ymin, ymax])
  #ax.plot(m1, m2, 'k.', markersize=2)
  #ax.set_xlim([xmin, xmax])
  #ax.set_ylim([ymin, ymax])
  #plt.show()

  # Regular grid to evaluate kde upon
#x_flat = np.r_[ent[:,0].min() : ent[:,0].max():128j]
#y_flat = np.r_[ent[:,1].min() : rvs[:,1].max():128j]
#xGrid = np.array([128, 128])
#y.np.shape = (128, 128)
#x,y = np.meshgrid(x_flat,y_flat)
#x, y = np.meshgrid(x, y)
#grid_coords = np.append(x.reshape(-1, 1), y.reshape(-1, 1), axis=1)

#z = kde(grid_coords.T)
#3z.reshape(128, 128)


#scatter(ent[:,0],ent[:,1])
##plt.show()
#imshow(z,aspect=x_flat.ptp()/y_flat.ptp())
#

import sys
sys.exit()
pdb.setTrace()

#################################################3
#--------------



#		WTS is points-col, itr-row
#fr=
x = np.random.random_integers(0, 9, 10000)
# corrupt half of output
y = x.copy()
indx = np.random.permutation(len(x))[:len(x) / 2]
y[indx] = np.random.random_integers(0, 9, len(x) / 2)

#for i in H[]:
#   H[i,:]=(f_i/fr)/F*log2(f_i/fr)
s = DiscreteSystem(x, (1, 10), y, (1, 10))
s.calculate_entropies(method='plugin', calc=['HX', 'HXY'])
#print( s.I() )


#Kernel
#Kernel.fast_kde(X[:,0] , X[:,1]    )




#print("*******HELLO fit2")

clf = BoostedClassifier(n_estimators=1, criterion=c)
clf.fit(iris.data, iris.target)
score = clf.score(iris.data, iris.target)
print(score)
print("*******SCORE")


print("sw=", clf.boost_method)
print("clf.fit", vars(clf.fit))
print("BoostedClassifier.fit", vars(BoostedClassifier.fit))
vars(BoostedClassifier)

print("atrr", getattr(clf, 'boost_weights'), clf.boost_weights)

#vars(BaseEnsemble)
#print("clf",clf.fit.WTS[3,:])
#print( clf.h2)

#def test_iris():
   # """Check consistency on dataset iris."""
for c in ("gini", "entropy"):
	# AdaBoost classification
	clf2 = BoostedClassifier(n_estimators=1, criterion=c)
 	clf2.fit(iris.data, iris.target)
	#print("clf2.fit.iris", vars(clf2.fit) )
	score = clf2.score(iris.data, iris.target)
	assert score > 0.9, "Failed with criterion %s and score = %f" % (c,
                                                                         score)

#print("clf2.fit.X", clf2.fit.X)
#print("***clf2.fit dict", clf2.fit(iris.data, iris.target).__dict__ , vars(clf2.fit) )
#print("atrr", getattr(clf2,'boost_weights'), clf2.boost_weights )
#print("hello2")


#class foo(obj)
