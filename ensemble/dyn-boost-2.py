#from weight_boosting.py
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
import numpy as np
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

    for iboost in xrange(BWB.n_estimators):
        # Boosting step
        sample_weight, estimator_weight, estimator_error = BWB._boost(
            iboost,
            X, y,
            sample_weight,
            X_argsorted=X_argsorted)

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



import csv

def parseNum(x):
    xx = x.replace(",", "")
    #if not xx.replace(".","").isdigit(): return x
    return "." in xx and float(xx) or int(xx)

dd=[('SYS', 'float64'),('DIA','float64'),('HR1','float64'),('OX','float64'),('HR2','float64'),('WHT','float64'),('Label',int)]
tele_raw_dta = np.recfromcsv("/home/solver/data/raw-labeled-th/all.csv", dtype=dd)

#x = [map(parseNum, line) for line in csv.reader(open("/home/solver/Desktop/data/smooth-mii/all.csv"))]
x = [map(parseNum, line) for line in csv.reader(open("/home/solver/data/raw-labeled-th/all.csv"))]
bst = np.asarray(x)
bstRows = bst.shape[0]
bstCol = bst.shape[1]; #print bstCol
patData = bst[:, 0:bstCol-1].copy()  #everything but last column(labels)
patTarg = bst[:, bstCol-1].copy()
print("l262, patdat pattar", patData.shape, patTarg.shape)

bwb=AdaBoostClassifier()
fit(bwb,patData,patTarg)
