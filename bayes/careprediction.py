"""
Estimation as likelihood to see a given alert among N number alerts with different alert rate sizes.  More likely to see alert with larger rate size.  In the simple case, determine probablility of seeing alert 60 our of single alert with size N.

To determine Probability distribution out of N alerts, take prior as log-uniform distribution, ie equal probability to see Nmax between 10^3-10^4, 10^4-10^5, ...  Each log interval therefore represents a alert type.

update to generate the posterior, take the mean to minimize the square error

now summarize the posterior as a credible interval-> cdf at 5%,95%

for probability model, get pmf for each one. then enumerate the possible pairs of values for each and calculate the distribution... DnD example p.54 thinkbayes.MakeMixture
"""

import thinkstats
import thinkplot
import thinkbayes
from thinkbayes import MakeGaussianPmf
from thinkbayes import Suite

datapr = {"ROULETTE": False, 
		"SUMMARY" : True,
		"POISSON" : False,
		"WEIBULL" : False,
		"POWER" : False}

class Alert( Suite ):
	def __init__(self, name=''):
		if datapr["SUMMARY"]:
			#use estimation procedure 
			# prior based on average of hard_alerts/total_alerts
			# or is is average of alerts/time ?
			if datapr["SUMMARY"]:
				mu = 3.0
				sigma = 0.5

			pmf = thinkbayes.MakeGaussianPmf(mu, sigma, 4)
			thinkbayes.Suite.__init__(self, pmf, name=name)

	def Likelihood(self, data, hypo):
		""" compute L( data under hypothesis) 
		hypo: alert rate as hard_alert/total_alert for some random interval
		data: hard alerts in a random interval
		
		"""
		#each hypothesis is possible value of lambda: long-term average number of alerts per that alert-type
		#distribution of alerts per interval is given by poisson pdf, distribution of time between alerts is exponential pdf 
		lam = hypo
		k = data
		like = thinkbayes.EvalPoissonPmf(lam, k)
		return like

		#if hypo < data:
		#	return 0
		#else:
			# log-likelihood

		#tuple exponential
		#x=hypo
		#hard,easy = data
		#like = x**hard * (1-x)**easy
		#return like
		# beta prior returns beta conjugate posterior
		# beta has two params (alpha,beta) if =1, then uniform 

#each val of lambda make a poisson pmf, and add to meta-pmf, then compute mixture
	def MakeAlertPmf(suite):
		metapmf = thinkbayes.Pmf()

		for lam, prob in suite.Items():
			pmf = thinkbayes.MakePoissonPmf(lam,10)
			metapmf.Set(pmf,prob)

		mix = thinkbayes.MakeMixture(metapmf)
		return mix

	#enumerated diffs
	#diff = diff1 - alert3_dist

	#sudden death_ time to alert
	def MakeAlertTimePmf(suite):
		metapmf = thinkbabyes.Pmf()

		for lam,prob in suite.Items():
			pmf = thinkbayes.MakeExponentialPmf(lam, high=2, n=2001)
			metapmf.Set(pmf, prob)
	
		mixT = thinkBayes.MakeMixture(metapmf)
		return mixT

	#average time between alert, interval variability
	#observer bias, more likely to occurr in larger interval (class-size, passenger on full airplane, why more likely to be on a plane that is full... wait time, longer, because more likely to arrive at a larger interval before next alert, where are the Hard alerts, if )
	def BiasPmf(pmf):
		new_pmf = pmf.Copy()
		
		for x,p in pmf.Items():
			new_pmf.Mult(x,x) #multiply prob(x) by the likelihood it will be observed
		new_pmf.Normalize()
		return new_pmf
	
#	does observe bias contribute to alert fatigue?
## predicting wait time to next hard alert?
#if seen 10 easy alerts, how long to wait until next hard alert
	#1.use z to compute prior zp(time as seen by observer)
	#2.use #passengers to estimate distribution of x, elapsed time since last train
	#3.y= zp -x to get distribution of actual wait time





# effect size
# evidence (sat)

#hiearchical



def main():
	#TH
	alert1 = Alert('top1')
	alert1.UpdateSet([8,7,3,5]) 
	#number of hardalerts of type1 for 4 intervals [3,5,2,1]
	alert2 = Alert('top2')
	alert2.UpdateSet([6,3,6,0])
	#alert3 = Alert('top3')
	#alert3.UpdateSet( [0,1,0,1] )
	
	#ICU
	alert1ICU = Alert('top1ICU')
	alert1ICU.UpdateSet([8,7,3,5]) #number of hardalerts of type1 for 4 intervals [3,5,2,1]
	alert2ICU = Alert('top2ICU')
	alert2ICU.UpdateSet([6,3,6,0])
	alert3ICU = Alert('top3ICU')
	alert3ICU.UpdateSet([0,1,1,1])

	thinkplot.Clf()
	thinkplot.PrePlot(num=2)
	thinkplot.Pmf(alert1)
	thinkplot.Pmf(alert2)
	thinkplot.Save(root='numberalerts',
					xlabel='Hard Alerts',
					ylabel='Probability',
					formats='pdf')


	#prob of alert-rate distribution
	alert1_dist = MakeAlertPmf(alert1)
	alert2_dist = MakeAlertPmf(alert2)
	alert3_dist = MakeAlertPmf(alert3)
	
	#time to alert
	time_dist1 = MakeGoalTimePmf(suite1)    
	time_dist2 = MakeGoalTimePmf(suite2)
	
	thinkplot.Clf()
	thinkplot.PrePlot(num=2)
	thinkplot.Pmf(time_dist1)
	thinkplot.Pmf(time_dist2)    
	thinkplot.Save(root='timeToAlert',
			xlabel='alert_rate: alerts to event',
          ylabel='Probability',
                 formats='pdf')

	diff = goal_dist1 - goal_dist2
	p_win = diff.ProbGreater(0)
	p_loss = diff.ProbLess(0)
	p_tie = diff.Prob(0)

main()





