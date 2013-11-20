# hard, easy on x-axis
# group by correct, incorrect

#from score import *
class rnkci(object):
	def __init__(self):
		self.hard = None
		self.easy = None
		self.inc = score.inc
		self.cor = None


from scipy import stats
import numpy as np
a = [1,2,3,15,34,55,66,7,8,9]

b= np.linspace(a.min(),a.max(),100)

kernel = stats.gaussian_kde(a)

pdf = kernel(b)
print 'pdf', pdf

#e =  [i.expect(kernel) for i in kernel(b)]

#e = [ i.expect(pdf) for i in a ]
pp = [ stats.norm.expect(kernel,p) for p in pdf]
print 'e', pp

#c = [True if(i<i.expect(kernel)) else False for i in a]
#print c

#print kernel(a)


#how assign probability to non-random variable? 
#assign probability as beliefs
#given r.v. E(r.v) = 1/_lambda
