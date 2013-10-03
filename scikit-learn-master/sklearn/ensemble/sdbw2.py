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
import csv, pprint
from scipy import stats
from matplotlib import pyplot as plt
from astroML.plotting import hist
from collections import Counter
from entrofunc import *

#Global variables to track the dynamic weights and correct/incorrect classifier
WTS = np.array([]); INCC = np.array([]);

### boost fit functions <br>
#1.def _boost_real
#2.def fit *IN USAGE
#3.def _boost_real_depr
#4.def fit_depr

### 1._boost_real	<br>
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

### 2._fit (INUSAGE)
# **IN-USAGE** boost fit  <br>
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
### decision function from ensemble/weight_boosting.py
def decision_function(BWB, X):
     """Compute the decision function of ``X``.

     Parameters
     ----------
     X : array-like of shape = [n_samples, n_features]
         The input samples.

     Returns
     -------
     score : array, shape = [n_samples, k]
         The decision function of the input samples. The order of
         outputs is the same of that of the `classes_` attribute.
         Binary classification is a special cases with ``k == 1``,
         otherwise ``k==n_classes``. For binary classification,
         values closer to -1 or 1 mean more like the first or second
         class in ``classes_``, respectively.
     """
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

### 3._boost_real_depr
def _boost_real_depr(BWB, iboost, X, y, sample_weight): #, X_argsorted=None):

	"""Implement a single boost using the SAMME.R real algorithm."""
    	# - global INC
    	INC = np.zeros((BWB.n_estimators, X.shape[0]),'f')
        estimator = BWB._make_estimator()

        if X_argsorted is not None:
            estimator.fit(X, y, sample_weight=sample_weight,
                          X_argsorted=X_argsorted)
        else:
            estimator.fit(X, y, sample_weight=sample_weight)

        y_predict_proba = estimator.predict_proba(X)

        if iboost == 0:
            BWB.classes_ = getattr(estimator, 'classes_', None)
            BWB.n_classes_ = len(self.classes_)

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

        return sample_weight, 1., estimator_error, incorrect


### 4._fit_depr()	<br>
def fit_depr(BWB, X, y, sample_weight=None):
    """Build a boosted classifier/regressor from the training set (X, y).
    Parameters
    X : array-like of shape = [n_samples, n_features]
    The training input samples.

    y : array-like of shape = [n_samples]
    The target values (integers that correspond to classes in
    classification, real numbers in regression).

    sample_weight : array-like of shape = [n_samples], optional
    Sample weights. If None, the sample weights are initialized to
    1 / n_samples.
    Returns
    BWB : object
    Returns BWB.
    """


    # - Check that the base estimator is a classifier
    if not isinstance(BWB.base_estimator, ClassifierMixin):
        raise TypeError("base_estimator must be a "
                            "subclass of ClassifierMixin")

    # - Check that algorithm is supported
    if BWB.algorithm not in ('SAMME', 'SAMME.R'):
        raise ValueError("algorithm %s is not supported"
                         % BWB.algorithm)

    #  - SAMME-R requires predict_proba-enabled base estimators
    if BWB.algorithm == 'SAMME.R':
        if not hasattr(BWB.base_estimator, 'predict_proba'):
            raise TypeError(
                "AdaBoostClassifier with algorithm='SAMME.R' requires "
                "that the weak learner supports the calculation of class "
                "probabilities with a predict_proba method.\n"
                "Please change the base estimator or set "
                "algorithm='SAMME' instead.")

    # - Check parameters
    if BWB.learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero")

    # - Check data
    X, y = check_arrays(X, y, sparse_format="dense")

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

    # - Create argsorted X for fast tree induction
    X_argsorted = None

    if isinstance(BWB.base_estimator, BaseDecisionTree):
        X_argsorted = np.asfortranarray(
            np.argsort(X.T, axis=1).astype(np.int32).T)
    X_argsorted2=X_argsorted
    global WTS
    WTS = np.zeros((BWB.n_estimators, X.shape[0]),'f')

    global INCC
    INCC = np.zeros(X.shape[0], 'f')

    for iboost in xrange(BWB.n_estimators):
        # - Boosting step
        sample_weight, estimator_weight, estimator_error = BWB._boost(
            iboost,
            X, y,
            sample_weight)
            #X_argsorted=X_argsorted)
        #t=[sample_weight2, val1float, estimator_error2, INCC]
        t = BWB._boost_real(
            iboost,
            X, y,
            sample_weight)
            #X_argsorted=X_argsorted2)

        # - Early termination
        if sample_weight is None:
            break

        BWB.estimator_weights_[iboost] = estimator_weight
        BWB.estimator_errors_[iboost] = estimator_error

        # - Stop if error is zero
        if estimator_error == 0:
            break

        sample_weight_sum = np.sum(sample_weight)

        #WTS[i,:] = sample_weight
        WTS[iboost,:] = t[0]
        if len(t)>2:
                INCC[iboost] = t[2]
        else:
                INCC[iboost] = 0
        # - Stop if the sum of sample weights has become non-positive
        if sample_weight_sum <= 0:
            break

        if iboost < BWB.n_estimators - 1:
            # - Normalize
            sample_weight /= sample_weight_sum


    return BWB

### DATA paths, input, labels <br>
# need to extend this with ponyORM
def parseNum(x):
    xx = x.replace(",", "")
    return "." in xx and float(xx) or int(xx)

dd=[('SYS', 'float64'),('DIA','float64'),('HR1','float64'),('OX','float64'),('HR2','float64'),('WHT','float64'),('Label',int)]
tele_raw_dta = np.recfromcsv("/home/solver/data/raw-labeled-th/all.csv", dtype=dd)
mimic_raw = np.recfromcsv("/home/solver/Desktop/data/raw-labled-mimic/all.csv", dtype=dd)

x = [map(parseNum, line) for line in csv.reader(open("/home/solver/Desktop/data/raw-labled-mimic/all.csv"))]
bst = np.asarray(x)
bstRows = bst.shape[0]
bstCol = bst.shape[1]; #print bstCol
patData = bst[:, 0:bstCol-1].copy()  #everything but last column(labels)
#labels
patTarg = bst[:, bstCol-1].copy()
print("l262, patdat pattar", patData.shape, patTarg.shape)

### BOOST <br>
bwb=AdaBoostClassifier()
fit(bwb,patData,patTarg)

### Score
# Determine correct incorrect calling decision function.
def wrong(decision,label):
	#returns true/false
	mark = decision != label
	incorrect = mark[mark==True]
	return incorrect

df=decision_function(bwb,patData)
df[df>0]=1.0
df[df<0]=-1.0

inc = wrong(df,patTarg)
print 'number incorrect = ' , len(inc)
print 'number correct = ' , len(patTarg) - len(inc)
pprint.pprint(inc)
pprint.pprint(df)

#fit_depr(bwb,patData,patTarg)
print 'inc:' , INCC
print 'error: ', bwb.estimator_errors_

#**ENTROPY** <br>
#BWB.n_estimators x sample-features <br>
print 'sample_weight ', bwb.estimator_weights_ , len(bwb.estimator_weights_)
entFeatures = np.asarray( WTS.copy() , dtype=np.float64)

entVal=np.array
#get the unique  weight vals and number
c = Counter()
for e in set(entFeatures.flat):
      c[e]+=1
print 'T_counts ', c.most_common(10), ' ', sum( c.values() )

#map, dont use xrange!<br>
#zip(*matrix) transposes the matrix row<=>col<br>
#entCnt returns the count based entropy<br>
entT = map(list, zip(*entFeatures))
ent = [entCnt( e ) for e in entT  ]
#write the entropy to file
#with open('entropy.csv', 'wb') as f:
#	writer = csv.writer(f)
#	for erow in ent: 
#		writer.writerows(erow) 
import pickle
entfile = open('entropy.pkl', 'wb')
#entfile.write(ent)
pickle.dump(ent,entfile)
#np.save(ent,entfile)
entfile.close

#numInc=[]
numInc = [i for i in INCC if i !=0]
numberIncorrect = len(numInc)
print 'INCC ' , INCC, numberIncorrect
print 'WTS ' , WTS.shape
print entFeatures.shape

# ** BAYESBLOCK binned-histogram
# http://www.astroml.org/examples/algorithms/plot_bayesian_blocks.html
print 'hello'
# PLOTS
#------------------------------------------------------------

# """
# 
#    There will be 3 plots with 3 factors within each (for tempo's sake).  The discussion involves the interaction between each factor, with each axis being generic but highly specific {patient-state, sequential-vs-static, form of data}.  The three plots are entropy, ranked-confidence-interval-label(fenwick-tree/AOR), and false-positive-curve.  Big question are what is hard vs easy?  What is global combinatoric health state look like statically or sequentially?  How can we best classify given noisy data, wide measures, sparse in some features, overfit in others? Can the form of the data be visualized?
# 
#   A ranked correct-incorrect-classification of each patient state gives dashboard-style overview of the engine(boost,entropy,false-positive), using a splom.  Look into fenwick trees or AOR to overlay.  What does a recurrence overlay look like?  Also, this is where multi-armed-bandit sampling can be applied.  As well as the kaggle overfitting-prevention method.  KDE smoothing provides some normalization of stationary vs non-stationary data.  How much data is lost, ie how much is dependent can be visualized in this stage as well.  Finally, there is the paper that auto-cross-validates.
# 
# 
#   The fpCurve relates accuracy to size, using 3 curves( window-size, kernel-size, and correct-incorrect(static)sequence-size(dynamic) ).  The interaction between window and kernel describes sequential-time.  The kernel-size-curve is given optimal-fixed window-size.  Kernel-size curve describes the possible patient state, that exist.  This can be enhanced with reccurrence plot.  Therefore define health as a recurrence state with persistent characteristics.  The interaction between CI/sequence and kernel-size further defines what does health mean.  Because we cannot simply say that health is defined by classification accuracy, which depends on occam's razor.  CI-boost-curve is given optimal-fixed kernel-size.  This  further describes the form of the data, and a recurrence plot that looks at stationary time-series features would be interesting.
# 
# 
# (threshold.py lists some of this)
# 1.return a rnkCIlabel -> correct incorrect for each pt-state	<br>
# 2.return a fpCurve(fp vs kernel-size..feature as bag-of-words) -> and do it for 3 curves	<br>
# window-size of smoothing, 	<br>
# <<<< sequential-time  >>>>>>	<br>
# kernel-size of kernel	<br>
# <<<<  form of the data >>>>>>	<br>
# ci-size(static)sequence-size(dynamic)	<br>
# 
# """

### bayes adaptive binning
# http://www.astroml.org/user_guide/density_estimation.html#bayesian-blocks-histograms-the-right-way
# The entropy measure is based on frequency count within a given measure.  If the width of the measure changese, so does the entropy and subsequent alert classification.  How to smooth a histogram thus is an important problem in identifying events.  Bayesian blocks addresses this problem using a fitness function to optimize bin-width, which are not required to be equal.  A smaller variation in bin width is consistent with uniform data distribution.  The bayesian likelihood function depends on block width and the number of points within a block.  For n number of points, there exist 2^n number of possible configurations.  Bayesian Blocks perform quadratically using a dynamic programming approach; the optimal (k+1)-th configuration is based on one of the k-th configurations.
#  To deal with some error, small random number around 0 can be added to prevent width from going to zero which causes problems with the log function, throws divide by zero error. 


# First figure: show normal histogram binning <br>

# bins=15 + bins=1000
# <IMG SRC="kn_by.png" ALT="img" WIDTH=500 HEIGHT=270>
fig = plt.figure(figsize=(10, 4))
fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15)

ax1 = fig.add_subplot(121)
ax1.hist(ent, bins=15, histtype='stepfilled', alpha=0.2, normed=True)
ax1.set_xlabel('entropy')
ax1.set_ylabel('P(entropy)')

ax2 = fig.add_subplot(122)
ax2.hist(ent, bins=1000, histtype='stepfilled', alpha=0.2, normed=True)
ax2.set_xlabel('entropy')
ax2.set_ylabel('P(entropy)')
#plt.savefig('normalhistogrambinning.png')
print 'fig1'

# ------------------------------------------------------------
# <IMG SRC="bayesblocks2.png" ALT="img" WIDTH=500 HEIGHT=270>
# Second & Third figure: Knuth bins & Bayesian Blocks
# have to set up virtualenv for astrolml and sklearn
# since bayesbinnexample.py works in the same module directory, I will simply export ent[] vector 

fig = plt.figure(figsize=(10, 4))
fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15)

for bins, title, subplot in zip(['knuth', 'blocks'],
                               ["Knuth's rule", 'Bayesian blocks'],
                               [121, 122]):
	ax = fig.add_subplot(subplot)

 	#plot a standard histogram in the background, with alpha transparency list
	hist(ent, bins=200, histtype='stepfilled', alpha=0.2, normed=True, label='standard histogram')

    # plot an adaptive-width histogram on top
   	hist(ent, bins=bins, ax=ax, color='black', histtype='step', normed=True, label=title)

   	ax.legend(prop=dict(size=12))
   	ax.set_xlabel('entropy')
   	ax.set_ylabel('P(entropy)')

#print 'fig2'
#plt.show()

### hard vs easy:


### stat significance
#wakari hot-hand
#sequential
#so distribution significance


### matplotlib
#plt.show
#plt.savefig('name.png')
