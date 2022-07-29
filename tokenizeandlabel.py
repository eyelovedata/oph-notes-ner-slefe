import pandas as pd
import re 
from nltk.tokenize.treebank import TreebankWordTokenizer
import numpy as np
import ast
import string 
import math
import time
from nltk.tokenize.treebank import TreebankWordTokenizer
from utils import gettokenspanlist, findlabel, findvaheader, findsleheader, findextheader, findllheader, findmatchingtoken, returnlabels, initializetokenlistlabels, gettokenlistlength, countmatches, remove_unicode_specials, MyLogger 

logger=MyLogger("logs/slefetokenizeandlabel.log") 

t = TreebankWordTokenizer()

inputfilepath="../data/slefe/slefelabels.csv"
outputfilepath="../data/slefe/slefetokenlabels.csv"
labelset = 'slefe'


def gettokenlabels(labelset, inputfilepath, outputfilepath, samplesize, startat): 
    '''
    labelset determines whether we want to do va NER or sle ner. One of four choices for now: ['va','sle', 'fe', 'slefe']
    '''
    start = time.time()
    logger.debug("reading input dataframe from "+ inputfilepath)
    
    if labelset == 'slefe':
        from slefelabelnames import labelnames
        df=pd.read_csv(inputfilepath, nrows=samplesize, skiprows=startat, header=None, 
        low_memory=True, names=['note_deid', 'extod', 'extos', 'sleodll', 'sleosll', 'sleodcs', 'sleoscs', 
        'sleodk', 'sleosk', 'sleodac', 'sleosac','sleodiris', 'sleosiris', 'sleodlens', 'sleoslens', 'sleodvit', 'sleosvit', 'feoddisc','feosdisc','feodcdr','feoscdr','feodmac','feosmac','feodvess','feosvess','feodperiph','feosperiph', 'DATE_OF_SERVICE',
                                'provider_deid', 'pat_deid', 'note'])
        del df["extod"]
        del df["extos"]

    df["note"]=df["note"].apply(remove_unicode_specials)
    logger.debug("length of input dataframe:" +str(len(df)))
    
    #keep rows where any labelnames are not null (drop rows with no labels extracted from sql)
    #saves a little time as we don't need to process these rows 
    df=df[df[labelnames].notnull().any(1)]
    
    print("tokenizing...")
    #tokenize
    t = TreebankWordTokenizer()
    df["tokens"]=df["note"].apply(t.tokenize)
    df["tokenspanlist"]=df["note"].apply(gettokenspanlist)
    df["doclength"]=df["tokenspanlist"].apply(len)
    
    #initialize tokenlist labels to all 'O'
    df["tokenlistlabels"]=df["tokenspanlist"].apply(initializetokenlistlabels)
    
    #for each label, find the spans, match-index, and update the token list labels
    print("finding the matches and updating the token list labels...")
    for name in labelnames: 
        df[name+'_spanlist']=df[[name,"note"]].apply(lambda x: findlabel(*x), axis=1)
        df[name+"_matchindexlist"]=df[["tokenspanlist",name+"_spanlist"]].apply(lambda x: findmatchingtoken(*x), axis=1) 
        df["tokenlistlabels"]=df[[name+"_matchindexlist","tokenlistlabels"]].apply(lambda x: returnlabels(*x, name), axis=1)
    
    logger.debug("length of resulting dataframe: "+ str(len(df)))
    
    #count how many smartform labels there are
    df['labelcount']=df[labelnames].count(axis=1)
    
    logger.debug("saving output dataframe to "+ outputfilepath)
    df[["note_deid", "doclength", "tokens", "tokenlistlabels", "labelcount"]].to_csv(outputfilepath, index=False)
    
    end = time.time()
    logger.debug("elapsed time: "+ str(end - start)+ " seconds")
    return 


logger.debug("reading back in the full dataset...") 

df=pd.read_csv(outputfilepath)

logger.debug("calculating ratio of matches to labels...") 

df["detectedlabelcount"]=df["tokenlistlabels"].apply(countmatches)
df["ratiodetectedlabels"]=df["detectedlabelcount"]/df["labelcount"]

logger.debug("length dataframe after discarding notes without complete label matches: "+ str(len(df[df["ratiodetectedlabels"]>=1]) ))

df[df["ratiodetectedlabels"]>=1].to_csv(outputfilepath[0:-4]+"-100pctlabelmatches.csv", index=False)
