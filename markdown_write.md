#smiley face in ipython
http://nbviewer.ipython.org/gist/deeplook/5162445
#heat map dendogram
http://nbviewer.ipython.org/github/ucsd-scientific-python/user-group/blob/master/presentations/20131016/hierarchical_clustering_heatmaps_gridspec.ipynb
#corr matrix to network graph MST
http://nbviewer.ipython.org/github/mvaz/PyData2014-Berlin/blob/master/3%20-%20DAX30%20Network.ipynb

#auto-tune hyperparameters, log-uniform search space
http://fastml.com/optimizing-hyperparams-with-hyperopt/
#trends in ml:
http://fastml.com/good-representations-distance-metric-learning-and-supervised-dimensionality-reduction/
#purpost of stats
http://slantedwindows.com/reservoir-sampling-made-visual/
http://rem.ph.ucla.edu//goal.of.stat.html
#health-analyis
#additive model: clinical study
http://www.ncbi.nlm.nih.gov/pubmed/19854637
extreme money-ball houston astros all in
http://www.theatlantic.com/health/archive/2014/08/how-racism-creeps-into-medicine/378618/?single_page=true
#ipython
http://nbviewer.ipython.org/urls/gist.githubusercontent.com/jdfreder/6734825/raw/529872f1b4544d6e270c92c0335c6a348ce6cf4a/IPython+Markdown+Pandoc+Limitations.ipynb
http://anandology.com/python-practice-book/iterators.html#generators
http://damon-is-a-geek.com/publication-ready-the-first-time-beautiful-reproducible-plots-with-matplotlib.html
http://wrobstory.github.io/2013/04/pandas-vincent-timeseries.html
http://nbviewer.ipython.org/gist/wrobstory/1eb8cb704a52d18b9ee8/Up%20and%20Down%20PyData%202014.ipynb
http://matplotlib.org/examples/api/radar_chart.html

#ml
https://bitbucket.org/mhorbal/learningtools
https://github.com/hmcuesta/PDA_Book

#latex
https://en.wikibooks.org/wiki/LaTeX/Mathematics

#d3
http://healthyalgorithms.com/2013/04/22/hello-world-of-statistical-graphics-in-ipython-notebook/
http://c3js.org/examples.html

#shedule pharm dosing
http://nbviewer.ipython.org/github/URXtech/techblog/blob/master/continuousTimeMarkovChain/markovChain.ipynb
from IPython.display import Image
i = Image(filename='ABC_MC.png')
i
%matplotlib inline




@DataSets
========

*refs*
http://www.healthmetricsandevaluation.org/news-events/seminar/overdiagnosed-making-people-sick-pursuit-health
http://www.businessweek.com/articles/2014-07-03/hospitals-are-mining-patients-credit-card-data-to-predict-who-will-get-sick

http://bugra.github.io/work/notes/2014-08-23/on-machine-learning/
http://wrobstory.github.io/2013/04/pandas-vincent-timeseries.html
http://jakevdp.github.io/blog/2014/06/10/is-seattle-really-seeing-an-uptick-in-cycling/
http://www.bayesimpact.org/blog/walking-the-beat.html
http://robjhyndman.com/hyndsight/aic/

http://stats.stackexchange.com/questions/31666/how-can-i-align-synchronize-two-signals

Purpose
-------
### Alert Fatigue
-----------------

### Machine Learning
--------------------
1. *what* is the data ? (unsupervised data)
2. *why* learning these things? (feature seletion)
3. *how* learning (optimization)

Machine learning is data and product dependent.  Successful, real-world application would require generalization from an input(training set), to unseen(test set) data.  The phases of machine learning are: transforming/representing the input, selecting a classifier, and evaluating appropriately.  

  First, real-world data is often not in useable form.  Data transforms may be from some input type to a vector form, such as bag-of-words and term-frequency from text, pixel values from images, size and width of bins from discrete data, etc.  Learning a better representation of the data, or even simply using more data, improves classifier accuracy more than complex models.  Data representation methods are unique to each problem and dependent on misclassification features.  Although, examples such as Vowpall rabbit, a linear classifier in large feature space, has proven successful, feature selection over every pixel combination of an image, does not reveal much for object recognition.  Therefore, exploring the features of misclassified observations may provide a pattern to improve data representation.  Finally, directly using domain knowledge about the data is important in building higher-order features.
  
  Selecting the best classifier involves understanding the strengths and weaknesses of the learning functions of each classifier, or choosing from a category to which the classifier belongs, such as deep learning, ensemble learning, graphical models, bayesian methods, etc.  Classifiers are more independent from input representation, but there are limits, such as noisy or limited data size, cost of error, the questions being asked, and metrics used in evaluation.  For example, weak, non-linear learners perform well with noisy data.  Error costs for a search engine, where both precision and recall to evaluate ranking are needed, are different from medical domain, which may put more emphasis on Type-I or Type-II error, depending on risk of oversensitivity vs. a failure in detection.  Situations where approximating a distribution vs individual observations is better suited, looking at KL-divergence can be helpful.  Finally, error costs in a classification problem(label-prediction) versus a regression problem(value-prediction) may not be equal.  
  
  Evaluating classifier accuracy involves specifying how it will learn.  Optimizing an estimator of the model to unseen data occurs at both the model and data levels; this helps to prevent poor generalization.  For example, situations with limited data or unbalanced class labels can lead to overfitting.  Selecting subsets of training vs test data, such as K-fold cross validation, for the former, or Stratified-K-fold cross validation, in the latter case, can prevent this.  Bagging, generating similar data with varying weights, can be used for data with label uncertainty, where errors cancel out in an aggregated, majority model.  Ensemble classifiers effect both the data and model space.  Finally, direct limits to the hypothesis space of a model can be used.  For example, in building a decision tree, the attribute that maximizes the information gain at each node is selected.  If a random subset of attributes is used to split a tree node, then a ensemble of subset trees forms a random forest.  


Curse of Dimensionality
----------------------
http://www.visiondummy.com/2014/04/curse-dimensionality-affect-classification/
The amount of training data needed to cover 20% of the feature range grows exponentially with the number of dimensions.

The three phases(what, why, and how) of machine learning refer to:  1.labeling what the data is, 2.identifying important features, and 3.optimizing the learning method used.  For example, unsupervised learning helps to understand what the data is; feature selection involves how learningwhy involves data transformation, model building, and evaluation


or engineered features(HoG, histogram of oriented features)

  [//] # (
Introduction:
-------------
Medical alerts have been shown to provide positive benefit: such as reduced blood sugar variability \cite{Mastro}, and improved prescribing safety \cite{Raebel}.  Currently, the extent of preventable medical error is estimated at 40,000 fatalities per year \cite{Muse}.  The need for improved medical alerts has been recognized in a recent Joint Commission and FDA statement.  The percentage of alerts currently being ignored, or alert fatigue, is estimated at 70-80\% \cite{Gouveia}.  The purpose of this study is to link model complexity with visual information for a clinically meaningful alert.  Recently, we were involved with a telehealth, at-home monitoring study, which recorded patient vital sign readings, twice daily, for an approximate several month interval.  A publicly available dataset is sampled as well for comparative analysis.  The dynamics of Adaboost weights have been shown to distinguish 'easy' from 'hard' classification points\cite{Capri2002}.  Data mining techniques are applied to handle features common to clinical data: high-dimensionality, variously scaled measures, group differences, and time series issues.  Three types of clinical alert are identified, static, sequential, and drifting.  Further, global monitoring over the combined feature space is achieved.  Finally, to address the issue of alert fatigue, results are graphically interpretable.  Decision tree, forest plot, and sequence logo are presented.




  )

### Workflow
make_file
hippa - bootstrapped data
   ![bigdata](files/consumerhealthhabits.png) 


### Data Description
	[//] # ( A note on Big Data Paradoxically, big data's predictive analytic problems are actually solved by relatively simple algorithms [2][4]. Thus we can argue that big data's prediction difficulty does not lie in the algorithm used, but instead on the computational difficulties of storage and execution on big data. (One should also consider Gelman's quote from above and ask "Do I really have big data?" ) The much more difficult analytic problems involve medium data and, especially troublesome, really small data. Using a similar argument as Gelman's above, if big data problems are big enough to be readily solved, then we should be more interested in the not-quite-big enough datasets.)   

<img src="files/consumerhealthhabits.png" height=350 width=200>
icons [linux],[pandas],[scikit],[ipython],[python],[d3]

Timeseries
----------
<--
http://stats.stackexchange.com/questions/10271/automatic-threshold-determination-for-anomaly-detection

find structure of time series:
such as intra-hour, intra-day

intervention detection:
anomalies- 1time(level shift) or systematic (trends)(clusters) number of trends 

transform:
 "residuals from a suitable model series" need to exhibit either a gaussian structure . This "gaussian structure" can usually obtained by incorporating one or more of the following "transformations" 1. an arima MODEL 2. Adjustments for Local Level Shifts or Local Time Trends or Seasonal Pulses or Ordinary Pulses 3. a weighted analysis exploiting proven variance heterogeneity 4. a possible power transformation ( logs etc ) to deal with a specific variance heterogenity 5. the detection of points in time where the model/parameters may have changed.



autocorrelated (inlier 1,9,1,9,1,9,5)
seasonal autoregressive data (say just monthof june is high)

 Tsays Journal of Forecasting article titled "Outliers, Level Shifts, and Variance Changes in Time Series " , Journal of Forecasting, Vol. 7, I-20 (1988).

*vincent
pandas http://nbviewer.ipython.org/github/nicolasfauchereau/NIWA_Python_seminars/blob/master/5_Introduction_to_Pandas.ipynb
jet-plot
  [//] # (refs/timeseriesregression.pdf  deals with autoregressive etc)
http://stats.stackexchange.com/questions/2077/how-to-make-a-time-seri|
sun spots
#http://nbviewer.ipython.org/gist/jhemann/4569783
http://stats.stackexchange.com/questions/31666/how-can-i-align-synchronize-two-signals
http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/TimeNormalization.ipynb
http://stats.stackexchange.com/questions/19715/why-does-a-time-series-have-to-be-stationary?rq=1

-->


## Clinical Study Design
-----------------------

  The gold standard of clinical study is double-blind, randoml controlled trial.  Post-study analysis, or sequential trial.  
  anova, effect size


Data Normalization
pooling, missing data, alignment
longitudinal, panel/cross-sectional
rct, double-blind
anova , effect size
http://stats.stackexchange.com/questions/90668/bayesian-analysis-of-contingency-tables-how-to-describe-effect-size?rq=1
http://nbviewer.ipython.org/github/dolaameng/tutorials/blob/master/sklearn-tutorials/PREPROCESSING%20-%20different%20data%20and%20methods.ipynb

https://github.com/nphoff/saxpy
http://fromstefanimportblog.blogspot.sg/2011/04/symbolic-aggregate-approximation-in.html
alittlebookfortimeseriesinr.pdf
http://www.stat.pitt.edu/stoffer/tsa3/  timseriesinr_yellowbook

http://www.statisticsdonewrong.com/
http://blog.mech.io/post/48631870585/using-cohort-analysis-to-make-your-funnels-more

## SAX symbolic aggregate approXimation
---------------------------------------
SAX transforms time-series data into symbolic strings.  The most common transform of raw time series data is Z-normalization (\mu=0, \sigma=1); it often preserves original time series features.  However, there are cases where this introduces error; for example when the values are mostly constant, with minor noise at short intervals, normalization will over-amplify the noise to its maximum.  Another technique used is piecewise aggregate approximation (PAA), which divides the original time series into $ M $ equally sized segments.  The mean for each segment is calculated, and the sequence assembled from each segments mean is the PAA transformation of the original time series.  PAA provides a lower bound, for the difference between the segments, set to the original Euclidean distances.  Finally, the $n$ values are transformed into strings of length $w<<n$, using an alphabet of size > 2.  The symbols produced correspond to time-series features with equal probability.  Using a lookup table, equal-sized areas from the normal curve, which the z-normalization transforms to, are selected.          





## Time Series Analysis
-----------------------
  **1. Preprocessing/Data Normalization
  ---------------------------------------------

  Dealing with data from different trials of different duration, which may contain missing data, requires resampling and alignment.  A simple method for time normalization is the percent method, where each trial is normalized to a percent cycle from 0 to 100%.  Interpolation ensures a fixed number of equally spaced data are generated.   
 **2. Correlations**
 -------------------
  Time series data refers to sequential observations ordered by time.  They may contain serial, periodic, or trend (daily,weekly) correlations.  Larger structural changes, such as mean-shift, mode-shift, or even a probability distribution change, are referred to as non-stationary.  If dependence between values are unaccounted for, this lead to overfitting of the data.  A stationary process is considered a stochastic process, whose joint probability distribution does not change when shifted in time.  Therefore, parameters such as mean and variance do not change over time or position; z-normalization of data (\mu =0, \sigma =1) is considered stationary.  An autocorrelation function tests for randomness at various lags for a timeseries; further the maximum lag can be determined to realign offset data.  A periodogram converts the autocorrelation from the time domain to frequency to determine cycle lengths.  
    
	**3. Short vs Long**
	 The use of short or long time ranges is dependent on model complexity, and vice versa.  Short data ranges refers to $\theta > observations$, and long data refers $observations >> \theta$.  To test a model, subsets of testing and training data are needed; for short data ranges there are not enough observations to allow a test subset.  Further, increases in the number of model parameters, noise, or randomness within the data itself, increase the size of data needed.  For short sample sizes, with increased model parameters or noise within the data, Akaike Information Criteria (AIC) is an in-sample measure for selecting models; however, it can only produce simple models with 1-2 parameters.  For intermediate series, when $obs > \theta$, parametric models may be appropriate.  Finally, for `obs >> 200`, differences between model and data are more apparent; non-parametric models, kernels based on time, or use of moving windows can be used.

the other one from the notebook
diff these two
 **1. Correlations**
 Time series data refers to sequential observations ordered by time.  They may contain serial, periodic, or trend (daily,weekly) correlations.  Larger structural changes, such as mean-shift, mode-shift, or even a probability distribution change, are referred to as non-stationary.  If dependencies are unaccounted for, they lead to overfitting of the data.  A stationary process is considered a stochastic process, whose joint probability distribution does not change when shifted in time.  Therefore, z-normalized data, with ( \mu =0, \sigma =1), is considered stationary.  
   
     **2. Short vs Long**
	 The use of short or long time ranges is dependent on model complexity, and vice versa.  Short data ranges refers to $\theta > observations$, and long data refers $observations >> \theta$.  To test a model, subsets of testing and training data are needed; for short data ranges there are not enough observations to allow a test subset.  Further, increases in the number of model parameters, noise, or randomness within the data itself, increase the size of data needed.  For short sample sizes, with increased model parameters or noise within the data, Akaike Information Criteria (AIC) is an in-sample measure for selecting models; however, it can only produce simple models with 1-2 parameters.  For intermediate series, when $obs > \theta$, parametric models may be appropriate.  Finally, for `obs >> 200`, differences between model and data are more apparent; non-parametric models, kernels based on time, or use of moving windows can be used.


Statistics
----------
extSdCard/medicalstatisticsStanford/
liptstick contains lead ? etc.

1. confounding factors: simpsons paradox, effect-size, group differences
2. fdr
3. frequentist vs bayesian



DataMining
----------
missing data, 
z-normalization
groups (bytefish.de)
pooling data, f-test of variances n>30 assume normal

## Time normalization
Pooling data across different trials with different duration requires an equal number of points be interpolated. 

Distribution
------------
### estimates

### kde, histogram
echen mixture non-param bayes

### log-normal

### goodness of fit
http://nbviewer.ipython.org/github/nicolasfauchereau/NIWA_Python_seminars/blob/master/4_Statistical_modelling.ipynb

### failure rate estimator
#healthyalgorithms.com/2014/05/16/mcmc-in-python-estimating-failure-rates-from-observed-data/



Exploratory Data analysis
-------------------------
sunspots : cleveland banking
http://ficolabsblog.fico.com/2013/01/a-moment-of-science-just-plot-it.html
seattle pd
bikes javkevpd
bugra moviedb analysis
frequencyplot website
trends?




@ALERTS
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
http://nbviewer.ipython.org/github/hildensia/bayesian_changepoint_detection/blob/master/Example%20Code.ipynb
http://stats.stackexchange.com/questions/492/dubious-use-of-signal-processing-principles-to-identify-a-trend?rq=1
http://stats.stackexchange.com/questions/31666/how-can-i-align-synchronize-two-signals
*notes

Signal and Outliers
-------------------
<!--
what is noise
what is normalization?
https://courses.edx.org/courses/RiceX/ELEC301x/T1_2014/courseware/7347598989364c4f994a39b7f755dffa/ee23e686563040efae8904aaf05102d0/
properties of signals
test signals

*occams razor, 
https://probmods.org/occam%27s-razor.html
size principle

outlierdetectiononlinesurvey.pdf
clustering k-nn, rank top clusters, etc


o utlier detection:
OFFLINE
1. productpartmodel ppm  hmm  ggm

windml -> visualize the clusters

ONLINE
A. so loewss sucks becuase does poorly at edges



1.use resistant smoothing window to remove nonstationary
tukey resistant smooth

window: narrow burst=3 small window of size 5 resist
        long : seasonality

2.rexpress so residuals are symmetric
3.apply control chart methods
http://stats.stackexchange.com/questions/1142/simple-algorithm-for-online-outlier-detection-of-a-generic-time-series?rq=1
quality control iqt tukey
median filter
long time series: autocorrelation and seasonaliy (recurring daily and weekend)

B. hyperloglog (geometric mean) bitpattern
bit-encode the online stream
http://research.neustar.biz/2012/10/25/sketch-of-the-day-hyperloglog-cornerstone-of-a-big-data-infrastructure/

transforms
http://stats.stackexchange.com/questions/10271/automatic-threshold-determination-for-anomaly-detection
-->

Signal, Noise, Randomness
-------------------------
  A signal is a description of how one parameter varies with another parameter. For instance, voltage changing over time in an electronic circuit, or brightness varying with distance in an image. A system is any process that produces an output signal in response to an input signal.




Discrete Fourier Transform
--------------------------
/*
http://andrew.gibiansky.com/blog/economics/accelerating-options-pricing-via-fourier-transforms/
http://www.brendangregg.com/FrequencyTrails/modes.html


*/
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
weibull cdf
#optimization: http://nbviewer.ipython.org/github/arokem/teach_optimization/blob/master/optimization.ipynb
#minimax:http://stackoverflow.com/questions/7856588/python-minimax-for-tictactoe 
#knapsack: http://bertolami.com/index.php?engine=blog&content=posts&detail=knapsack-problem
http://math.stackexchange.com/questions/103608/what-does-the-fourier-transform-mean-in-the-context-of-images
'''

 
##	http://www.r-bloggers.com/a-kernel-density-approach-to-outlier-detection/
Kernel Regression
-----------------
Given set of observations $$(x_i,y_i)$, with $x_i=(x_i1 , ... , x_ip)^T \epsilon \mathbf{R}^p,$$
$$\hat{f}_\min argmin_{a_0}\Sigma_i\kappa \left\frac{x-x_i}{h}\right (y_i - \textit{P_n}(x_i))^2$$
Where \textsl{P_n} is a polynomial of order \textit{n}, constant term \textit{a_0}, \kappa is kernel to weight values, and \textit{h} is bandwidth.  A local-polynomial smoothing, using Taylor-decomposition, in 1D:
$$\hat{f}_\min argmin_{a_0}\Sigma_i\kappa \left\frac{x-x_i}{h}\right \lefty_i - a_0 - a_1(x-x_1) - ... - a_n\frac{(x-x_i)^n}{n!}\right^2 $$

Nonparametric representation has greater flexibility than parametric, which are restricted to model and parameters. Basic linear regression assumes independence between error terms; this condition is not true for time series data, which contain high temporal dependency. To construct a model based on what is suggested by the data requires greater amounts of data.  Using higher order polynomials, or smaller kernel bandwidths with small data size, leads to overfitting. The bandwidth of the kernel is the standard deviation; it is chosen to be small for large, tightly packed data, and large for sparse, small data sets. Bootstrapping can compute the confidence intervals from the estimates; values outside the confidence interval are labeled outliers.   





Bayes Changepoint outlier detection
-----------------------------------
	''' OFFLINE: get log-probability, Pcp, take exp. sum get p(t_i, is_changepoint)
		input:: 
			1. prior of successive[a=cp,b=cp] at t_distance
		2. likelihood_data:[s_sequence, t_distance] no changepoint 
	ONLINE: gives prob_distribution(mass) of P(t) not_cp in [1,2,...n]; n=0 is P(t) is changepoint
	'''
1. time series factors mu, sigma, order(autoregression), cross-correlation
2. log likelihood, so its max is max of fn(obs)
   marginal likelihood 'sampling distribution'

3.segmentation, ppm, hmm,  
  GGM structure gives what?
4.online
survey of methods
http://research.neustar.biz/2012/10/25/sketch-of-the-day-hyperloglog-cornerstone-of-a-big-data-infrastructure/
http://stats.stackexchange.com/questions/1142/simple-algorithm-for-online-outlier-detection-of-a-generic-time-series?rq=1

Change point detection can identify changes in dependency structure in times series.  In general, it determines whether any change has occured, or if several changes have occured.  Using a bayesian framework, the problem becomes formulated as $$\textit{p(posterior) \propto  p(likelihood)p(prior)}$$

correlation coefficient
http://andrew.gibiansky.com/blog/machine-learning/speech-recognition-neural-networks/


Group Comparison
----------------
anova
radar plot
euclidean distance: 
#http://nbviewer.ipython.org/github/carljv/Will_it_Python/blob/master/MLFH/ch9/ch9.ipynb

!! hddm
http://ski.clps.brown.edu/hddm_docs/tutorial_python.html#within-subject-effects
dealing with outlier
generate data with outlier, but fit a stat model without taking outliers into account

use mixture model, assume outlir come from uniform, capture majority of trials




@BOOSTING
=========

*refs*
http://bugra.github.io/work/notes/2014-05-16/entropy-perplexity-image-text/
diversity index/perplexity
http://nbviewer.ipython.org/github/pprett/pydata-gbrt-tutorial/blob/master/gbrt-tutorial.ipynb


Ensemble Learning
----------------
weights



model complexity
--------------
###deviance plot, overfit
###crossValidation
###f-test, bic/aic
#balance dataset
#cost-sensitive learning p147 MLinAction


autonlab > infogain11.pdf
Entropy
-------
IG = H(Y) - H(Y|X)
relative IG = IG / H(Y)
use IG to determine how interesting a 2d-contingency table would be..
ie predict if live past 80y/o:
  ig(long life | smoker), ig(long life| gender), ...
  what is contingency table ?


Weights
-------


sequential
----------
waldboost
mab(kl-diverge, thompson sampling -> bayes control (causality)  )
2005	The Max K- Armed Bandit: A New Model of Exploration Applied to Search Heuristic Selection	Vincent A. Cicirello, Drexel University
Stephen F. Smith, Carnegie Mellon University
reservoir sampling
secretary problem









@Disease Modeling
================
sampling-optimization
---------------- -------- --------
-(vowpal rabbit) linear model over all features
-random forest
#http://blog.echen.me/2011/03/14/laymans-introduction-to-random-forests/
http://aeon.co/magazine/philosophy/is-the-most-rational-choice-the-random-one/
http://www.analyticbridge.com/forum/topics/challenge-of-the-week-random-numbers
-reservoir
#http://gregable.com/2007/10/reservoir-sampling.html
-secretary problem
(datagenetics)
http://datagenetics.com/blog/december32012/index.html
-thompson sampling !w
-small sample method
http://stats.stackexchange.com/questions/1856/application-of-machine-learning-techniques-in-small-sample-clinical-studies
logistic regression/l1 (chernoff)
# lasso regression of rank for patient based on alert-term fr
# http://nbviewer.ipython.org/github/carljv/Will_it_Python/blob/master/MLFH/ch6/ch6.ipynb
-model selection

#lda
http://blog.echen.me/2011/08/22/introduction-to-latent-dirichlet-allocation/
http://wellecks.wordpress.com/2014/09/04/these-are-your-tweets-on-lda-part-i/
wordle


#dirchlet
http://www.drbunsen.org/gambling-in-multiplayer-games/
http://nbviewer.ipython.org/github/dolaameng/tutorials/blob/master/sklearn-tutorials/SAMPLING%20GMM%20v.s.%20Dirichlet%20Process.ipynb

#graphical model
http://nbviewer.ipython.org/github/dolaameng/tutorials/blob/master/sklearn-tutorials/PRACTICE%20-%20time%20series%20%2B%20covariance%20%2B%20clustering.ipynb
GraphLassoCV
http://unsupervisedlearning.wordpress.com/2014/08/04/topological-anomaly-detection/
percolation clique

#mab
#http://blog.yhathq.com/posts/the-beer-bandit.html
#bayes mab (ukrainian seattle data science co.)
#http://cstheory.stackexchange.com/questions/21338/how-much-time-to-recognize-palindromes-in-logarithmic-space
http://blog.vctr.me/monty-hall/

#--- hypothesis model ---#
# ridge regression mlh ch4

#-disease model-- 
#http://blog.yhathq.com/posts/predicting-customer-churn-with-sklearn.html

optimization
grid search parameter space -> NIMS newzealend fish guy does it

sample size - funnel graph


*refs*
random choice:
http://www.keithschwarz.com/darts-dice-coins/
http://aeon.co/magazine/philosophy/is-the-most-rational-choice-the-random-one/
http://bugra.github.io/work/notes/2014-04-06/graphs-databases-and-graphlab/
http://sociograph.blogspot.com/
http://unsupervisedlearning.wordpress.com/2014/08/04/topological-anomaly-detection/
http://nbviewer.ipython.org/github/nicolasfauchereau/NIWA_Python_seminars/blob/master/4_Statistical_modelling.ipynb

measure theory
--------------
define metrics for health?
volume of taco
dynamic time warping: http://nbviewer.ipython.org/github/dolaameng/tutorials/blob/master/sklearn-tutorials/FEATURE%20-%20dynamic%20time%20warping%20for%20images.ipynb
euclidean distance
http://blog.echen.me/2011/03/14/laymans-introduction-to-measure-theory/

curse of dim.
simpsons paradox

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


@Bayes
=====
*refs*
http://connor-johnson.com/2014/02/18/linear-regression-with-python/
hierarchical 

very good link !
*** drift diffusion: http://ski.clps.brown.edu/hddm_docs/abstract.html

effect size: http://stats.stackexchange.com/questions/90668/bayesian-analysis-of-contingency-tables-how-to-describe-effect-size?rq=1
anova std of finite pop and superpop, make box-plot

<!--
likelihood functions:
A.estimator ('burst number', german tank, window-size(roc) )

B.time to (survival/censorying, churn)

C.encoding(obs.likelihood, hyperloglog bitpatter)


*notes*
  [//] # (

  kruscke diagram

occamns razor: admissibility dutch book
https://probmods.org/occam%27s-razor.html

  )
-->
within suject dependies
predictive posterior check (uncertainty of distribution)
http://ski.clps.brown.edu/hddm_docs/tutorial_python.html#within-subject-effects

MCMC outlier detection (unpooled)
----------------------
model the signal
uniform if unknown, or use failure rate estimates of prior if know
http://ski.clps.brown.edu/hddm_docs/methods.html#sequential-sampling-models
bugra

pooled hierarchy model
----------------------
http://twiecki.github.io/blog/2014/03/17/bayesian-glms-3/


propensity model
---------------
echen
gwern (berkson paradax)
nielsen (judea perl, theorem of causality)


@References
----------
A surge of p - values between 0.040 and 0.049 in recent decades ( but negative results are increasing rapidly too) Winter, Dodou, U of Delft
Half of a coin: negative probabilities, Gabor Szekely

searching and Mining Trillions of Time Series Subsequences under Dynamic Time Warping	Thanawin Rakthanmanon, University of California Riverside; et al.

@AppendixI: Math/algo Concepts
===
1. Fast Fourier Transform 
symmetric
jakevdp explains it
2. decompose
bugra gaussian process model
3. limits of adaboost bound  : chernoff distribution
4. random (sample from cdf, dice)
http://www.keithschwarz.com/darts-dice-coins/
5. entropy (dice), youtube video
6. german tank problem
7. concat, merge, join, cartesian product
   top5 searches that you used 
   database joins
On the complexity of division and set joins in the relational algebra	Dirk Leinders & Jan Van den Bussche, Limburgs Universitair Centrum
Worst-case Optimal Join Algorithms	Hung Q. Ngo, University at Buffalo; et al.
Efficient Dissection of Composite Problems, with Applications to Cryptanalysis, Knapsacks, and Combinatorial Search Problems Itai Dinur 1 , Orr Dunkelman Nathan Keller , and Adi Sham, 217.pdf

8. dynamic bayesian binning
ghost -> publish ideas blog
https://ghost.org/
