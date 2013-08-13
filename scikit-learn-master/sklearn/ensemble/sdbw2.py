#from weight_boosting.py
#from sklearn.ensemble import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.utils import array2d,check_arrays, check_random_state, column_or_1d
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree.tree import BaseDecisionTree
from sklearn.base import ClassifierMixin, RegressorMixin, BaseEstimator
from sklearn.tree._tree import DTYPE
#", "ExtraTreeClassifier","ExtraTreeRegressor"]
#from ..utils import array2d, check_arrays, check_random_state, column_or_1d
import numpy as np
from entrofunc import *
WTS = np.array([]); INCC = np.array([]);

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


	# Instances incorrectly classified

	incorrect = y_predict != y

    # Error fraction

	estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))

    # Stop if classification is perfect

	if estimator_error <= 0:
            return sample_weight, 1., 0.

    # Construct y coding as described in Zhu et al [2]:
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

    # Displace zero probabilities so the log is defined.
    # Also fix negative elements which may occur with
    # negative sample weights.

	y_predict_proba[y_predict_proba <= 0] = 1e-5

    # Boost weight using multi-class AdaBoost SAMME.R alg

	estimator_weight = (-1. * BWB.learning_rate
                                * (((n_classes - 1.) / n_classes) *
                                   inner1d(y_coding, np.log(y_predict_proba))))

    # Only boost the weights if it will fit again

	if not iboost == BWB.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(estimator_weight *
                                    ((sample_weight > 0) |
                                     (estimator_weight < 0)))
	return sample_weight, 1., estimator_error


def fit(BWB, X, y, sample_weight=None):
        """Build a boosted classifier/regressor from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The training input samples.

        y : array-like of shape = [n_samples]
            The target values (integers that correspond to classes in
            classification, real numbers in regression).

        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check parameters
        if BWB.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        # Check data
        X, y = check_arrays(X, y, sparse_format="dense")

        y = column_or_1d(y, warn=True)

        if ((getattr(X, "dtype", None) != DTYPE) or
                (X.ndim != 2) or (not X.flags.contiguous)):
            X = np.ascontiguousarray(array2d(X), dtype=DTYPE)

        if sample_weight is None:
            # Initialize weights to 1 / n_samples
            sample_weight = np.empty(X.shape[0], dtype=np.float)
            sample_weight[:] = 1. / X.shape[0]
        else:
            # Normalize existing weights
            sample_weight = np.copy(sample_weight) / sample_weight.sum()

            # Check that the sample weights sum is positive
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        # Clear any previous fit results
        BWB.estimators_ = []
        BWB.estimator_weights_ = np.zeros(BWB.n_estimators, dtype=np.float)
        BWB.estimator_errors_ = np.ones(BWB.n_estimators, dtype=np.float)

        for iboost in xrange(BWB.n_estimators):
            # Boosting step
            sample_weight, estimator_weight, estimator_error = BWB._boost(
                iboost,
                X, y,
                sample_weight)

            # Early termination
            if sample_weight is None:
                break

            BWB.estimator_weights_[iboost] = estimator_weight
            BWB.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if iboost < BWB.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum

        return BWB
def _boost_real_depr(BWB, iboost, X, y, sample_weight): #, X_argsorted=None):

	"""Implement a single boost using the SAMME.R real algorithm."""
    	#global INC
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

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        # Construct y coding as described in Zhu et al [2]:
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

        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        y_predict_proba[y_predict_proba <= 0] = 1e-5

        # Boost weight using multi-class AdaBoost SAMME.R alg
        estimator_weight = (-1. * BWB.learning_rate
                                * (((n_classes - 1.) / n_classes) *
                                   inner1d(y_coding, np.log(y_predict_proba))))

        # Only boost the weights if it will fit again
        if not iboost == BWB.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(estimator_weight *
                                    ((sample_weight > 0) |
                                     (estimator_weight < 0)))

        return sample_weight, 1., estimator_error, incorrect



def fit_depr(BWB, X, y, sample_weight=None):
    """Build a boosted classifier/regressor from the training set (X, y).

    Parameters
    ----------
    X : array-like of shape = [n_samples, n_features]
    The training input samples.

    y : array-like of shape = [n_samples]
    The target values (integers that correspond to classes in
    classification, real numbers in regression).

    sample_weight : array-like of shape = [n_samples], optional
    Sample weights. If None, the sample weights are initialized to
    1 / n_samples.

    Returns
    -------
    BWB : object
    Returns BWB.
    """


    # Check that the base estimator is a classifier
    if not isinstance(BWB.base_estimator, ClassifierMixin):
        raise TypeError("base_estimator must be a "
                            "subclass of ClassifierMixin")

    # Check that algorithm is supported
    if BWB.algorithm not in ('SAMME', 'SAMME.R'):
        raise ValueError("algorithm %s is not supported"
                         % BWB.algorithm)

    #  SAMME-R requires predict_proba-enabled base estimators
    if BWB.algorithm == 'SAMME.R':
        if not hasattr(BWB.base_estimator, 'predict_proba'):
            raise TypeError(
                "AdaBoostClassifier with algorithm='SAMME.R' requires "
                "that the weak learner supports the calculation of class "
                "probabilities with a predict_proba method.\n"
                "Please change the base estimator or set "
                "algorithm='SAMME' instead.")

    # Check parameters
    if BWB.learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero")

    # Check data
    X, y = check_arrays(X, y, sparse_format="dense")

    if sample_weight is None:
        # Initialize weights to 1 / n_samples
        sample_weight = np.empty(X.shape[0], dtype=np.float)
        sample_weight[:] = 1. / X.shape[0]
    else:
        # Normalize existing weights
        sample_weight = np.copy(sample_weight) / sample_weight.sum()

        # Check that the sample weights sum is positive
        if sample_weight.sum() <= 0:
            raise ValueError(
                "Attempting to fit with a non-positive "
                "weighted number of samples.")

    # Clear any previous fit results
    BWB.estimators_ = []
    BWB.estimator_weights_ = np.zeros(BWB.n_estimators, dtype=np.float)
    BWB.estimator_errors_ = np.ones(BWB.n_estimators, dtype=np.float)

    # Create argsorted X for fast tree induction
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
        # Boosting step
        sample_weight, estimator_weight, estimator_error = BWB._boost(
            iboost,
            X, y,
            sample_weight,
            X_argsorted=X_argsorted)
        #t=[sample_weight2, val1float, estimator_error2, INCC]
        t = BWB._boost_real(
            iboost,
            X, y,
            sample_weight,
            X_argsorted=X_argsorted2)

        # Early termination
        if sample_weight is None:
            break

        BWB.estimator_weights_[iboost] = estimator_weight
        BWB.estimator_errors_[iboost] = estimator_error

        # Stop if error is zero
        if estimator_error == 0:
            break

        sample_weight_sum = np.sum(sample_weight)

        #WTS[i,:] = sample_weight
        WTS[iboost,:] = t[0]
        if len(t)>2:
                INCC[iboost] = t[2]
        else:
                INCC[iboost] = 0
        # Stop if the sum of sample weights has become non-positive
        if sample_weight_sum <= 0:
            break

        if iboost < BWB.n_estimators - 1:
            # Normalize
            sample_weight /= sample_weight_sum


    return BWB

#INCC, WTS

import csv

def parseNum(x):
    xx = x.replace(",", "")
    #if not xx.replace(".","").isdigit(): return x
    return "." in xx and float(xx) or int(xx)

dd=[('SYS', 'float64'),('DIA','float64'),('HR1','float64'),('OX','float64'),('HR2','float64'),('WHT','float64'),('Label',int)]
tele_raw_dta = np.recfromcsv("/home/solver/data/raw-labeled-th/all.csv", dtype=dd)
mimic_raw = np.recfromcsv("/home/solver/Desktop/data/raw-labled-mimic/all.csv", dtype=dd)
#DATA-INPUT
#x = [map(parseNum, line) for line in csv.reader(open("/home/solver/Desktop/data/smooth-mii/all.csv"))]
#x = [map(parseNum, line) for line in csv.reader(open("/home/solver/data/raw-labeled-th/all.csv"))]
x = [map(parseNum, line) for line in csv.reader(open("/home/solver/Desktop/data/raw-labled-mimic/all.csv"))]
bst = np.asarray(x)
bstRows = bst.shape[0]
bstCol = bst.shape[1]; #print bstCol
patData = bst[:, 0:bstCol-1].copy()  #everything but last column(labels)
patTarg = bst[:, bstCol-1].copy()
print("l262, patdat pattar", patData.shape, patTarg.shape)

#BOOST
bwb=AdaBoostClassifier()
fit(bwb,patData,patTarg)
print 'error: ', bwb.estimator_errors_
print 'itr ', bwb.n_estimators

##ENTROPY BWB.n_estimators x sample-features
print 'sample_weight ', bwb.estimator_weights_ , len(bwb.estimator_weights_)
#entFeatures = np.array( (bwb.n_estimators , bstRows), 'float64' )
#print 'est-weights ', entFeatures.shape, bwb.estimator_weights_.shape
entFeatures = np.asarray( WTS.copy() , dtype=np.float64)
print 'entFeat (wts-copy)' , entFeatures.shape[0], entFeatures

entVal=np.array
#get the unique  weight vals and number
from collections import Counter
c = Counter()
for e in set(entFeatures.flat):
        c[e]+=1
print 'counts ', c
#print 'weight counts ',c[0:2]
print 'T_counts ', c.most_common(10), ' ', sum( c.values() )
#for i in c:
#        if i ==
entFeatures.flat
print entFeatures.shape[0]
#if entFeatures.shape[1] and ent = np.zeros( entFeatures.shape[1] , 'float64' )
for i in xrange( entFeatures.shape[1] ):
       #ent[i]= entroNumpy( entFeatures[:,i] )
       ent[i]= entCnt( entFeatures[:,i] )
print 'ent ',ent, len(ent)
#print 'wht-row-test ', entFeatures[0,:]
ent2 = np.zeros( entFeatures.shape[0] , 'float64' )
for j in xrange( entFeatures.shape[0] ):
        ent2[j] = entroNumpy( entFeatures[j,:])
#print 'ent2 ',ent2,len(ent2)

entCnt=[]
entCnt = Counter(ent)
#print 'entropy counts ', entCnt
#print 'entropy row-wise counts ', Counter(ent2)
#entVal=np.bincount(ent)
#entCnt = np.nonzero(entVal)

numInc=[]
numInc = [i for i in INCC if i !=0]
numberIncorrect = len(numInc)
#print 'INCC ' , INCC, numberIncorrect
#print 'WTS ' , WTS

#bayes-block binned-histogram
#http://www.astroml.org/examples/algorithms/plot_bayesian_blocks.html

import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

from astroML.plotting import hist

# draw a set of variables
#np.random.seed(0)
#t = np.concatenate([stats.cauchy(-5, 1.8).rvs(500),
#                    stats.cauchy(-4, 0.8).rvs(2000),
#                    stats.cauchy(-1, 0.3).rvs(500),
#                    stats.cauchy(2, 0.8).rvs(1000),
#                    stats.cauchy(4, 1.5).rvs(500)])
#
## truncate values to a reasonable range
#t = t[(t > -15) & (t < 15)]

#------------------------------------------------------------
# First figure: show normal histogram binning
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

#------------------------------------------------------------
# Second & Third figure: Knuth bins & Bayesian Blocks
fig = plt.figure(figsize=(10, 4))
fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15)

for bins, title, subplot in zip(['knuth', 'blocks'],
                                ["Knuth's rule", 'Bayesian blocks'],
                                [121, 122]):
    ax = fig.add_subplot(subplot)

    # plot a standard histogram in the background, with alpha transparency
    hist(ent, bins=200, histtype='stepfilled',
         alpha=0.2, normed=True, label='standard histogram')

    # plot an adaptive-width histogram on top
    hist(ent, bins=bins, ax=ax, color='black',
         histtype='step', normed=True, label=title)

    ax.legend(prop=dict(size=12))
    ax.set_xlabel('WTS')
    ax.set_ylabel('P(WTS)')

plt.show()
