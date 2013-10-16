#TODO	Sun 13 Oct 2013 05:17:47 PM CDT
#bubble wrap from visual this book to show weight dynamics


# boost wrapper and driver, data is pickled

#TODO	Thu 03 Oct 2013 06:30:53 PM CDT
#make the file locations different for different run and 
#maybe need a boostdriver.py


"""

<script type="text/javascript" src="jquery-latest.min.js"></script>
<link href="knowlstyle.css" rel="stylesheet" type="text/css" />
<script type="text/javascript" src="knowl.js"></script>

"""



### modules
# `cd ~/project/scikit-learn-master` <br>
#  `python -m sklearn.ensemble.sdbw2.py`	<br>
# DONE astroML does not seem to work.	2013-09-26 13:03

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.utils import array2d,check_arrays, check_random_state, column_or_1d
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree.tree import BaseDecisionTree
from sklearn.base import ClassifierMixin, RegressorMixin, BaseEstimator
from sklearn.tree._tree import DTYPE
from .weight_boosting import _samme_proba

import numpy as np
import csv, pprint, pickle
from scipy import stats
from matplotlib import pyplot as plt
from astroML.plotting import hist
from collections import Counter
from entrofunc import *



### wrapper functions	<br>
# Cannot access the classes or subclass.  
# Therefore copy needed functions and pass not self but the class it is borrowing from.
# A common pattern.


### single boost estimator
def _boost_real(BWB, iboost, X, y, sample_weight):
	"""Implement a single boost using the SAMME.R real algorithm."""
	INC = np.zeros((BWB.n_estimators, X.shape[0]),'f')
	estimator = BWB._make_estimator()

	try:
		estimator.set_params(random_state=BWB.random_state)
	except ValueError:
		pass

	estimator.fit(X, y, sample_weight=sample_weight)

	y_predict_proba = estimator.predict_proba(X)

	if iboost == 0:
            BWB.classes_ = getattr(estimator, 'classes_', None)
            BWB.n_classes_ = len(BWB.classes_)


	y_predict = BWB.classes_.take(np.argmax(y_predict_proba, axis=1),
                                       axis=0)


	# - Instances incorrectly classified

	incorrect = y_predict != y

    # - Error fraction

	estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))

    # - Stop if classification is perfect

	if estimator_error <= 0:
            return sample_weight, 1., 0.

    # - Construct y coding as described in Zhu et al [2]:
    #
    #    y_k = 1 if c == k else -1 / (K - 1)
    #
    # where K == n_classes_ and c, k in [0, K) are indices along the second
    # axis of the y coding with c being the index corresponding to the true
    # class label.

	n_classes = BWB.n_classes_

	classes = BWB.classes_

	y_codes = np.array([-1. / (n_classes - 1), 1.])

	y_coding = y_codes.take(classes == y[:, np.newaxis])

    # - Displace zero probabilities so the log is defined.
    # Also fix negative elements which may occur with
    # negative sample weights.

	y_predict_proba[y_predict_proba <= 0] = 1e-5

    # - Boost weight using multi-class AdaBoost SAMME.R alg

	estimator_weight = (-1. * BWB.learning_rate
                                * (((n_classes - 1.) / n_classes) *
                                   inner1d(y_coding, np.log(y_predict_proba))))

    # - Only boost the weights if it will fit again

	if not iboost == BWB.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(estimator_weight *
                                    ((sample_weight > 0) |
                                     (estimator_weight < 0)))
	return sample_weight, 1., estimator_error


### fit runs over estimators 
def fit(BWB, X, y, sample_weight=None):
        """
		Build a boosted classifier/regressor from the training set (X, y).
        Parameters*
        X : array-like of shape = [n_samples, n_features]
            The training input samples.

        y : array-like of shape = [n_samples]
            The target values (integers that correspond to classes in
            classification, real numbers in regression).

        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.
        Returns*
        self : object
            Returns self.
        """
        # - Check parameters
        if BWB.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        # - Check data
        X, y = check_arrays(X, y, sparse_format="dense")

        y = column_or_1d(y, warn=True)

        if ((getattr(X, "dtype", None) != DTYPE) or
                (X.ndim != 2) or (not X.flags.contiguous)):
            X = np.ascontiguousarray(array2d(X), dtype=DTYPE)

        if sample_weight is None:
            # - Initialize weights to 1 / n_samples
            sample_weight = np.empty(X.shape[0], dtype=np.float)
            sample_weight[:] = 1. / X.shape[0]
        else:
            # - Normalize existing weights
            sample_weight = np.copy(sample_weight) / sample_weight.sum()

            # - Check that the sample weights sum is positive
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        # - Clear any previous fit results
        BWB.estimators_ = []
        BWB.estimator_weights_ = np.zeros(BWB.n_estimators, dtype=np.float)
        BWB.estimator_errors_ = np.ones(BWB.n_estimators, dtype=np.float)
        global WTS
        WTS = np.zeros((BWB.n_estimators, X.shape[0]),'f')

        for iboost in xrange(BWB.n_estimators):
            # - Boosting step
            sample_weight, estimator_weight, estimator_error = BWB._boost(
                iboost,
                X, y,
                sample_weight)

            # - Early termination
            if sample_weight is None:
                break

            BWB.estimator_weights_[iboost] = estimator_weight
            BWB.estimator_errors_[iboost] = estimator_error
            WTS[iboost]=sample_weight

            # - Stop if error is zero
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)
            # - Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if iboost < BWB.n_estimators - 1:
                # - Normalize
                sample_weight /= sample_weight_sum

        return BWB


### decision function 
# from ensemble/weight_boosting.py
def decision_function(BWB, X):
#     """Compute the decision function of ``X``.
#
#     Parameters
#     ----------
#     X : array-like of shape = [n_samples, n_features]
#         The input samples.
#
#     Returns
#     -------
#     score : array, shape = [n_samples, k]
#         The decision function of the input samples. The order of
#         outputs is the same of that of the `classes_` attribute.
#         Binary classification is a special cases with ``k == 1``,
#         otherwise ``k==n_classes``. For binary classification,
#         values closer to -1 or 1 mean more like the first or second
#         class in ``classes_``, respectively.
#     """
     BWB._check_fitted()
     X = np.asarray(X)

     n_classes = BWB.n_classes_
     classes = BWB.classes_[:, np.newaxis]
     pred = None

     if BWB.algorithm == 'SAMME.R':
         # The weights are all 1. for SAMME.R
         pred = sum(_samme_proba(estimator, n_classes, X)
                    for estimator in BWB.estimators_)
     else:   # BWB.algorithm == "SAMME"
         pred = sum((estimator.predict(X) == classes).T * w
                    for estimator, w in zip(BWB.estimators_,
                                            BWB.estimator_weights_))

     pred /= BWB.estimator_weights_.sum()
     if n_classes == 2:
         pred[:, 0] *= -1
         return pred.sum(axis=1)
     return pred

### load data
def parseNum(x):
    xx = x.replace(",", "")
    return "." in xx and float(xx) or int(xx)

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


### boost function call <br>
# insert article writeup here
bwb=AdaBoostClassifier()
fit(bwb,patData,patTarg)


### score.. 
# see pandadata.py	<br>
# function to tally incorrect=1, correct=0	<br>
# decision function built-in to determine correct incorrect	<br>
def incArray(decision,label):
	incArr = [ 1 if(z[0] != z[1]) else 0 for z in zip(decision,label) ] 
	c = Counter()
	for i in incArr:
		c[i]+=1
	print "no incorrect", c[1]
	print "no correct", c[0]
	return incArr #full length

df=decision_function(bwb,patData)
df[df>0]=1.0
df[df<0]=-1.0
inc = incArray(df,patTarg)



### pkl
import pickle
# weight.pkl
wtfile = open('weight.pkl', 'wb')
pickle.dump(WTS,wtfile)
wtfile.close

# incorrect.pkl
incfile = open('incorrect.pkl', 'wb')
pickle.dump(inc,incfile)
incfile.close

# entropybasic.pkl
entFeatures = np.asarray( WTS.copy() , dtype=np.float64)
c = Counter()
for e in set(entFeatures.flat):      #get the unique  weight vals and number
	c[e]+=1
entT = map(list, zip(*entFeatures))  #zip(*matrix) transposes the matrix row<=>col<br>
ent = [entCnt( e ) for e in entT  ]  #entCnt returns the count based entropy<br>

entfile = open('entropybasic.pkl', 'wb')
pickle.dump(ent,entfile)
entfile.close

# databasic.pkl
vitals = dd
alerts =  np.asarray( patTarg.copy() , dtype=np.float64)
readings = np.asarray( patData.copy() , dtype=np.float64) 
dt = [ vitals, alerts, readings ]

entfile = open('datahdrlbl.pkl', 'wb')
pickle.dump(dt,entfile)
entfile.close


### sampling
# simpsons paradox
#phantom correlation
##n a general sense (the proof being left as an exercise for the reader):
#
#    Given two measurements xi in X and yi in Y on a set of points p1…n in P, if the value of xi+yi increases the chance that pi will be sampled, it will introduce a phantom correlation between X and -Y
#
	#Kind of scary, eh?
#http://wordpress.euri.ca/2012/youre-probably-polluting-your-statistics-more-than-you-think/

### ROC score CROSS-VALIDATE DATA
##TODO	Thu 03 Oct 2013 12:49:35 PM CDT
# http://stats.stackexchange.com/questions/10271/automatic-threshold-determination-for-anomaly-detection?rq=1  use roc score to set threshold on max tp min fn
# prie: system to improve rule-set ROC score
