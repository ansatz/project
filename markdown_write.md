DataSets
========

*refs*
http://wrobstory.github.io/2013/04/pandas-vincent-timeseries.html
http://jakevdp.github.io/blog/2014/06/10/is-seattle-really-seeing-an-uptick-in-cycling/

Log-Norm Distribution
--------------------


Distribution
------------
http://www.bayesimpact.org/blog/walking-the-beat.html
kde, histogram


Timeseries
----------
*vincent



Alerts
======
*univariate, input representation*

*refs*
*fft
http://archive.today/ulPFkfferings
http://www.dspguide.com/ch9/3.htm
music.pdf benson musical offerings
http://cs.stackexchange.com/questions/12497/what-the-difference-between-the-fourier-transform-of-an-image-and-an-image-histo
#http://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/
Multiplying signals, amplitude modulation
http://dsp.stackexchange.com/questions/7527/fft-time-domain-average-vs-frequency-bin-average
*krg
http://bugra.github.io/work/notes/2014-05-11/robust-regression-and-outlier-detection-via-gaussian-processes/
http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gp_regression.html#example-gaussian-process-plot-gp-regression-py


*mcmc
mcmc http://bugra.github.io/work/notes/2014-04-26/outlier-detection-markov-chain-monte-carlo-via-pymc/
*bycp

*notes

Signal, System, Functions
-------------------------

A signal is a description of how one parameter varies with another parameter. For instance, voltage changing over time in an electronic circuit, or brightness varying with distance in an image. A system is any process that produces an output signal in response to an input signal.



Discrete Fourier Transform
--------------------------
parametric
  The Discrete Fourier Transform(DFT) deals with decomposing a signal into a sum of sinusoid functions of different frequencies.

###Forward Discrete Fourier Transform:

$$X_k = \sum_{n=0}^{N-1}x_n\cdot e^\frac{-i 2\pi k n}{N}$$

###Inverse Discrete Fourier Transform (IDFT):
$$ x_n = \sum_{k=0}^{N-1}X_k e^ \frac{-i 2\pi k n}{N}$$


  The transform from $x_n\rightarrow X_k$  represents the translation from time domain to frequency domain.  The observed sequence is a convolution of complex, periodic functions with a finite windowing function. The frequency spectrum identified depends on the windowing size; in a plot of frequency vs time, a smaller window has more localised time but greater spread in frequency, while larger window sizes better identify true frequency components but are spread more in the time axis.  If observation times are not integer multiples of the period, some inchorent sampling occurs.  Finally, aperiodic changes, over time, within $$x_n$$ make frequency ranges harder to identify.

    Convolution in the time domain corresponds to multiplication in the frequency domain; however, using DFT is faster and working in the frequency domain easier for interpretation.  The frequency response corresponds to the amplitude and phase changes of the cosine waves used in decomposing the signal.  These parameters fully describe a linear system, which contains additive signals that can be separated.  Fourier transform is robust to outlier detection; reconstruction of the original signal from the frequency domain(IDFT) does not require amplitude or variance of the original signal.  This has advantage over mean-averaging and median-filter, which change the original signal or can introduces false-positives, in regions with small changes, respectively. 

	Ensemble averaging in the frequency domain is used to pool individual patient data.

	optimization
	    Therefore, outlier detection using fourier transform requires window size, frequency amplitude, and frequency threshold parameters to be set.  If the frequency response of the signal contains a component greater than the frequency threshold, then the position is considered an outlier. 

'''
#optimization: http://nbviewer.ipython.org/github/arokem/teach_optimization/blob/master/optimization.ipynb
#minimax:http://stackoverflow.com/questions/7856588/python-minimax-for-tictactoe 
#knapsack: http://bertolami.com/index.php?engine=blog&content=posts&detail=knapsack-problem
'''

 
Kernel Regression
-----------------
Given set of observations $$(x_i,y_i)$, with $x_i=(x_i1 , ... , x_ip)^T \epsilon \mathbf{R}^p,$$
$$\hat{f}_\min argmin_{a_0}\Sigma_i\kappa \left\frac{x-x_i}{h}\right (y_i - \textit{P_n}(x_i))^2$$
Where \textsl{P_n} is a polynomial of order \textit{n}, constant term \textit{a_0}, \kappa is kernel to weight values, and \textit{h} is bandwidth.  A local-polynomial smoothing, using Taylor-decomposition, in 1D:
$$\hat{f}_\min argmin_{a_0}\Sigma_i\kappa \left\frac{x-x_i}{h}\right \lefty_i - a_0 - a_1(x-x_1) - ... - a_n\frac{(x-x_i)^n}{n!}\right^2 $$

Nonparametric representation has greater flexibility than parametric, which are restricted to model and parameters. Basic linear regression assumes independence between error terms; this condition is not true for time series data, which contain high temporal dependency. To construct a model based on what is suggested by the data requires greater amounts of data.  Using higher order polynomials, or smaller kernel bandwidths with small data size, leads to overfitting. The bandwidth of the kernel is the standard deviation; it is chosen to be small for large, tightly packed data, and large for sparse, small data sets. Bootstrapping can compute the confidence intervals from the estimates; values outside the confidence interval are labeled outliers.   






Bayes Changepoint outlier detection
-----------------------------------


Boosting
========

*refs*
http://bugra.github.io/work/notes/2014-05-16/entropy-perplexity-image-text/
diversity index/perplexity
http://nbviewer.ipython.org/github/pprett/pydata-gbrt-tutorial/blob/master/gbrt-tutorial.ipynb
Ensemble Learning
----------------

model complexity
--------------
###deviance plot, overfit
###crossValidation
###f-test, bic/aic


Entropy
-------

sampling-optimization
--------
reservoir
grid search parameter space

sample size - funnel graph


Disease Modeling
================
*multivariate, feature engineering*

*refs*
http://bugra.github.io/work/notes/2014-04-06/graphs-databases-and-graphlab/
http://sociograph.blogspot.com/
http://unsupervisedlearning.wordpress.com/2014/08/04/topological-anomaly-detection/
http://nbviewer.ipython.org/github/nicolasfauchereau/NIWA_Python_seminars/blob/master/4_Statistical_modelling.ipynb

ols
---

bagging
-------


network
--------
clique
graph db
topo


sequential
----------
wald
mixture

semidefinite programming
------------------------


Bayes
=====
*refs*
http://connor-johnson.com/2014/02/18/linear-regression-with-python/

MCMC outlier detection (unpooled)
----------------------
model the signal
uniform if unknown, or use failure rate estimates of prior if know

pooled hierarchy model
----------------------
http://twiecki.github.io/blog/2014/03/17/bayesian-glms-3/






