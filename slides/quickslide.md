# notes:


- 	pandoc -s -S -i -t slidy --mathjax quickslide.md -o quickslide.html --self-contained

	pandoc -t slidy -s slides -o view
	docs pycco -w *.py

# data
-- Telehealth Readings/Day , %Change_Level

-<img src="./quickslides/seabornplots/census_th_factor_deepblue.png" alt='img'> 

-- ICU patients

-<img src="./quickslides/seabornplots/census_mmc_jointscatter.png" alt='img'> 

-- mimic patient expire

-<img src="./quickslides/seabornplots/pmfdiff.png" alt='img'> 

-- demographic information

<img src="./quickslides/seabornplots/boxplotgeoagegender.png" alt='img'> 

-- log-normal distribution

<img src="./quickslides/seabornplots/lognormaldistribution.png" alt='img'> 

-- categorical data

weekday trend

<img src="../plots/mimic/maleicuexpiredvitalspt.png" alt="img" width="500" height="600">

# boosting:
(needs to be rerun over full dataset)
-- weights: 

-Weights/Iteration of Boosting
<img src="./slidefigs/wts.png" alt='img'> 

-- Hard/Easy/Correct/Incorrect plot:

<img src="./slidefigs/factorplotHEIC_m.png" alt='img'> 

-- logistic regression

<img src="./quickslides/alrtfr.png" alt='img'> 
<img src="./quickslides/alrtspc.png" alt='img'> 

# alerts
## need to be rerun
-- bootstrap ci (resample %_method, shown invariant to data normalization issues)

<img src="./quickslides/ci.png" alt='img'> 

-- kernel regression

<img src="./quickslides/slidefigs/kernelregression.png" alt='img'> 


-- FFT
<img src="./quickslides/seabornplots/fft_hr1_1000pts.png" alt='img'> 



-- Bayes Change Point
<img src="./quickslides/seabornplots/bayescp_hrsubject_id15.png" alt='img'> 


# bayes
-- prediction of care = P(#alert_distribution) + P(time_between_alerts)_weibull_prior + P(#alerts_unseen)

-- effect size (number of patients needed)

-- roulette method
www.sr32	
-- possible models:

 <img src="slidefigs/kruscke.jpg" alt="img" width="500" height="270">
 design likelihood function 
 P(INTV) - probability(intervention)=distribution_over_alerts + distribution_time_between

causal modeling (Judea Perl)

# code + data
-- github.com/ansatz/project/

-- telehealth:
 I can deidentify and bootstrap the data, so the statistics are the same(mean, var) but the actual data is not shared.

-- MIMIC2:

 https://mimic2app.csail.mit.edu/
When registerig list Dr.Avitall. He will be contacted and then db access is given.






