### refs
#http://www.gwern.net/Lewis%20meditation
#http://www.gwern.net/LSD%20microdosing
#http://blog.wolframalpha.com/2013/10/10/gotta-compute-em-all-wolframalphas-new-data-about-pokemon/
#http://www.economist.com/news/briefing/21588057-scientists-think-science-self-correcting-alarming-degree-it-not-trouble
#http://www.economist.com/news/briefing/21588057-scientists-think-science-self-correcting-alarming-degree-it-not-trouble
#http://www.statsblogs.com/2013/10/20/how-do-you-write-your-model-definitions/

###
http://blog.nextgenetics.net/?e=94
take hypergeometric distribution (n choose k)
20k genes =T 400 in group a
1000 genes differentiate, 300 are in group a
so if pick 1000 randomly what chance that more than 300 are cell cycle
cell cycle =T, not cell =F binomial
so 20k choose 2  => 200K ways to choose, how many TT, TF, FF ?
TT equals the 400 choose 2.  = 80k
TF equals  (400 choose 1) * (19.6k choose 1) 
FF equals 16.6K choose 2

so that was for binomial ... whatabout k=300
so take sum of all combinations from P(300<no of x= T<1000, no of y=F<700)

### p-values
Probability observe data as far from null (significance), given the null=true.  It is not the P(null=T) | observation.  P(a|b) != P(b|a)
just because data unlikely under assumption(null) != evidence that null is true.
evidence for null does not equal failure to find evidence against
power analysis: ability to correctly reject null, if goodness-of-fit high stat power=> interpret high p-val as evidence real relationship (NULL) not very likely

### models and stat. methods
do not describe stat method by how to calculate but by what model is assumed and fitted.
(linear regression is calculatd with ordinary least squares) to calculate best fit line to some data
better to see ols as computational method used to fit linear model, assuming normally distributed residuals
how to write probablity model?
1. describe all non-stochastic + error terms
2. start stochastic then non-stochastic (ie a more prob.distribution centric)
3. graphical a.BUGS b. kruschke(model distribution inherent) quick and easy

### adaptive modeling : significance tests
http://vserver1.cscs.lsa.umich.edu/~crshalizi/weblog/
1. novel test stat with null distr. being a asymptote
2. tukey outlier detection (dimensions of anomaly)
quartiles: q1 - 1.5qr  q3 + 1.5qr (qr ==  q3-q1)

### big data asymptote
#alogrithmic weakening
http://arxiv.org/pdf/1309.7804v1.pdf

Michael I. Jordan165, "On the Computational and Statistical Interface and 'Big Data'"166 (special joint statistics/ML seminar)
    Abstract: The rapid growth in the size and scope of datasets in science and technology has created a need for novel foundational perspectives on data analysis that blend the statistical and computational sciences. That classical perspectives from these fields are not adequate to address emerging problems in "Big Data" is apparent from their sharply divergent nature at an elementary level---in computer science, the growth of the number of data points is a source of "complexity" that must be tamed via algorithms or hardware, whereas in statistics, the growth of the number of data points is a source of "simplicity" in that inferences are generally stronger and asymptotic results can be invoked. Indeed, if data are a data analyst's principal resource, why should more data be burdensome in some sense? Shouldn't it be possible to exploit the increasing inferential strength of data at scale to keep computational complexity at bay? I present three research vignettes that pursue this theme, the first involving the deployment of resampling methods such as the bootstrap on parallel and distributed computing platforms, the second involving large-scale matrix completion, and the third introducing a methodology of "algorithmic weakening," whereby hierarchies of convex relaxations are used to control statistical risk as data accrue. 

wide vs tall MLforHckText

### meaningful
statistical space vs biological space: enrichment study
meaningless (time-series data)

### causal modeling
http://www.statsblogs.com/2013/10/20/proving-causation/
1.experimental design: to prove 1 factor cause another, need randomised experiment, where every variable that can have effect is randomised.
2.generalizability: prove in population sample must be random
3.observational study: (chance encounters wild seber)


### micromort


### time-of-day effects


### day-of-week effect
#holidays, 

# validity:
	When n=1 , internal validity may apply, but how to apply that to outside cases refers to systematic 

# power: 
	effect size is large(noticeable) then a small experiment, but want to control for mutliple comparisons(lower p-val in one-tailed test)

Assumption are large effect size, p=0.01, paired/within subject experimental design.
library(pwr)
pwr.t.test(d=0.8, type=paired, power=0.8, alternative=greater, sig-level=0.1)
paired t power calc n=18.874

19pairs means 19active +19placebo
but assuming strong effect size(0.8), if cut in half medium-effect (d=0.4) then need 66 pairs or 132 days.  if d is oversold, really is 0.4, then power to detect effect reduces from 80% to 24%.  trying to debug, say bug is not there(ie the null is not there), that is a type2 error, 

how treat days(3):?


#family wise error rate:  bonferonni 1/n.  false positive (type1 error) reject the null(what stat logic is based on) hypothesis as number of hypothesis increase then rare event increases.. increase the false positive rate.  type2 error is prove the null, say its not there when it is.
bonferonni assumes independent data.  so lose alot of data.  benjamin and hochberg1995 have a sequential t-test for each feature, sorted p-value.  if want false discovery rate to be .05 (q-value), then test from last to first p-value if less than (current index*fdr/n). stop when get first inequality and call rest the hypothesis significant.

#bayes beats the t-test
Bayesian estimation for 2 groups provides complete distributions of credible values for the effect size,
group means and their difference, standard deviations and their difference, and the normality of the data.
The method handles outliers. The decision rule can accept the null value (unlike traditional t tests) when
certainty in the estimate is high (unlike Bayesian model comparison using Bayes factors). The method
also yields precise estimates of statistical power for various research goals. The software and programs
are free and run on Macintosh, Windows, and Linux platforms.
http://www.indiana.edu/~kruschke/BEST/
https://github.com/strawlab/best

#AOR odds ratio

#positive pred val, neg pred val


# validity:
	When n=1 , internal validity may apply, but how to apply that to outside cases refers to systematic 

# meijer briggs (not valid but reproducible)



#summary stats
#jitter
