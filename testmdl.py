from sdbw2 import *

bwbr=AdaBoostClassifier()

tele_raw_dta = np.recfromcsv("/home/solver/data/raw-labeled-th/all.csv", dtype=dd)
mimic_raw = np.recfromcsv("/home/solver/Desktop/data/raw-labled-mimic/all.csv", dtype=dd)

x = [map(parseNum, line) for line in csv.reader(open("/home/solver/Desktop/data/raw-labled-mimic/all.csv"))]
bst = np.asarray(x)

bstRows = bst.shape[0]                                                                              
bstCol = bst.shape[1]; #print bstCol                                                                
patData = bst[:, 0:bstCol-1].copy()  #everything but last column(labels)                            
patTarg = bst[:, bstCol-1].copy()
print("l262, patdat pattar", patData.shape, patTarg.shape)                                          


