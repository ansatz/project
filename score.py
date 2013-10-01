### score object
"""

A score class object 
  data:[ [a,b,c], [],  ] (nx6)
  entropy:[            ] (nx1)
  kde:[                ] (nx1) if 1 is hard 0 easy
  bin:[                ] (nx1) if 1 is hard 0 easy

"""

#
class score(object):
	def __init__(self, datafile, entropyfile):
		#data-files
		dd=[('SYS', 'float64'),('DIA','float64'),('HR1','float64'),('OX','float64'),('HR2','float64'),('WHT','float64'),('Label',int)]
		tele_raw_dta = np.recfromcsv("/home/solver/data/raw-labeled-th/all.csv", dtype=dd)
		mimic_raw = np.recfromcsv("/home/solver/Desktop/data/raw-labled-mimic/all.csv", dtype=dd)
		
		#entropy
		entfile=open('/home/solver/project/scikit-learn-master/sklearn/ensemble/entropy.pkl','rb')
		entpkl = pickle.load(entfile)
		entnp = np.asarray( entpkl, dtype=np.float64)
		entfile.close()

		#members
		self.dataT = tele_raw_data
		self.dataM = mimic_raw
		#self.entr =  entropyfile
		self.entropy = entnp

	def histo(self):
		#------------------------------------------------------------
		# First figure: show normal histogram binning
		fig = plt.figure(figsize=(10, 4))
		fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15)

		ax1 = fig.add_subplot(121)
		ax1.hist(self.entropy, bins=15, histtype='stepfilled', alpha=0.2, normed=True)
		ax1.set_xlabel('entropy bins=15')
		ax1.set_ylabel('Count(t)')

		ax2 = fig.add_subplot(122)
		ax2.hist(self.entropy, bins=200, histtype='stepfilled', alpha=0.2, normed=True)
		ax2.set_xlabel('entropy bins=200')
		ax2.set_ylabel('Count(t)')

		#------------------------------------------------------------
		# Second & Third figure: Knuth bins & Bayesian Blocks
		fig = plt.figure(figsize=(10, 4))
		fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15)
		
		for bins, title, subplot in zip(['knuth', 'blocks'],
		                                ["Knuth's rule-fixed bin-width", 'Bayesian blocks variable width'],
		                                [121, 122]):
		    ax = fig.add_subplot(subplot)
		
		    # plot a standard histogram in the background, with alpha transparency
		    hist(self.entropy, bins=200, histtype='stepfilled',
		         alpha=0.2, normed=True, label='standard histogram')
		
		    # plot an adaptive-width histogram on top
		    hist(self.entropy, bins='blocks', ax=ax, color='black',
		         histtype='step', normed=True, label=title)
		
		    ax.legend(prop=dict(size=12))
		    ax.set_xlabel('entropy bins')
		    ax.set_ylabel('C(t)')
		
		plt.show()

	def smooth(self):


	def comparePlot(self):
		#hard vs easy
		#correct vs incorrect


### Get correct and incorrect values


### 
