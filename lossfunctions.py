###REFS
http://nbviewer.ipython.org/urls/raw.github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/master/Chapter5_LossFunctions/LossFunctions.ipynb
https://github.com/johnmyleswhite/BanditsBook
levin
oreillya bandits book 
ch7 ml in action manning
jakulin05thesis.pdf

# Taleb- 'better to be vaguely right than very wrong'
Stress payoff of decisions not accuracy.

### Loss function
"""
- Defined as function of true parameter and estimate of the parameter.
L(_theta,_thetahat) = f(_theta,_thetahat)

- squared-loss: measures fit of estimate, greater the loss, greater the difference
	assymetric(over perhaps preferable to under estimate), 
	absolute magnitude of differences linearized vs outlier can influence loss quadratically)
- zero-one loss if estimate not equal then 1
  log_loss =  -_thlog(_t) - (1-_th)log(1- _t)

- design a loss function focus on outcomes


"""
# philosophy
Data consists of instances and attributes.  Models can be thought of as the meaning of the data.  Loss function judge model quality.  teh choice of loss functions and model families are factorizations of probability distributions.  

# beyond entropy
differential entropy can be negative.  Mutual information and Pearson correlation coefficient 

# exploratory analysis:
Based on mutual information and interaction.  Use Rajkis distance to stat quantities to distances.  Then cluster both attributes and their values.  Look at matrix to see interaction between pairs.  Explore concept drift and non-random patterns of missing data. Does statistical significance of interaction associated with whether an expert found them meaningful in the medical domain.  


# mab


# network analysis (interaction models)
kikuchi bayes classifier based on approximate fusion of marginals with kikuchi method.  It performs greedy search for best structure in terms of interactions.  Uses parsimous prior and model averaging. 

parts-as-constraints, loglinear models, max entropy
kirkwood superpostiions


