import pandas as pd
import random 
from utils import MyLogger, checkfornomatches 

logger=MyLogger("logs/slefetraintestsplit.log")


#removes rows with no labels 
#performs 80/10/10 split of dataset into train/val/test sets, splitting by note 

inputfilepath="../data/slefe/slefetokenlabels-100pctlabelmatches.csv"
outputfilepath="../data/slefe/"

logger.debug("reading data from: "+inputfilepath)

dftokenlabels=pd.read_csv(inputfilepath)

logger.debug("number of notes in input:"+str(len(dftokenlabels)))

logger.debug("now doing the data spliting...") 

idx = list(dftokenlabels["note_deid"].unique())

import random
random.seed(1)
random.shuffle(idx) 
valsize=round(0.1*len(idx))
testsize=round(0.1*len(idx))
validx=idx[0:valsize]
testidx=idx[valsize:valsize+testsize]
trainidx=idx[valsize+testsize:]

logger.debug("N notes in train set: "+str(len(trainidx))+", saved to " + outputfilepath+"train.csv")
logger.debug("N notes in val set: "+str(len(validx))+", saved to " + outputfilepath+"val.csv")
logger.debug("N notes in test set :"+str(len(testidx))+", saved to " + outputfilepath+"test.csv")

dftokenlabels[dftokenlabels["note_deid"].isin(trainidx)].to_csv(outputfilepath+"train.csv", index=False)
dftokenlabels[dftokenlabels["note_deid"].isin(testidx)].to_csv(outputfilepath+"test.csv", index=False)
dftokenlabels[dftokenlabels["note_deid"].isin(validx)].to_csv(outputfilepath+"val.csv", index=False)


