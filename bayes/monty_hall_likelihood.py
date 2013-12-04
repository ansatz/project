"""
input:
1. dr.monty 
There is an alert(static), this is event1. Then event2 is the choice to cancel based on whether hard/easy.  'Choice' is max posterior update.
	
2. mcmc 'hot-hand-streak'
Difference of means given alert2, given alert1.

3. disease model 'fp-tree' association map
Run fp-tree, reset labels, then run mcmc to see if significant difference from static or hot-hand-2-alert.

output:
update posterior

FRAMEWORK:
hypothesis:	prior|posterior : likelihood

1. estimate prior for alert
[ locomotive problem -> mean of posterior ]
with different bounds the estimate changes. so use different datasets and it will converge
graph it with different distributions (power, weishart, uniform)
credible_interval -> cdf 95%
page30 (select prior)

power law distribution: look at frequency of alerts, see if they fit that

so estimate as probability of seeing alert among many alerts:: ie the likelihood is greater for larger alert(sets)


2. (label-boost): estimate prior for hard-easy
3. what is the likelihood for hard|alert
4. update posterior


"""

from thinkbayes import Suite

	

class Alert(Suite):
	def __init__(self,hypos):
		Pmf.__init__(self)
		for hypo in hypos:
			self.Set(hypo,1)
		self.Normalize()
	

	hypoA = dict(alert
	hypoB = dict(

	hypos = 'HE'
	pmf = Alert(hypos)


	def Likelihood(self,data,hypo):
		if hypo==data:



def main():
		






