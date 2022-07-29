import sys
sys.path.append('C:/Users/Sophia/Documents/ResearchPHI/STRIDE_FULL/oph-notes-ner/oph-notes-ner/')

import pandas as pd
import re 
from nltk.tokenize.treebank import TreebankWordTokenizer
import numpy as np
import string 
import math
import time
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from utils import gettokenspanlist, findlabel, findmatchingtoken, returnlabels, initializetokenlistlabels, gettokenlistlength, checkfornomatches, remove_unicode_specials, MyLogger, select_longest_regex_finding

"""## Regular Expressions"""
sleregex=r'(?i)(((\s(eye)?l(ids)?(\/|&|(\sand\s))?l(ashes)?)).*\s+((c(onjunctiva)?(\/|&|(\sand\s))?s(clera)?)).*\s+(k|(cornea)).*\s+a(nterior)?\s?c(hamber)?.*\s+(iris).*\s+(lens).*\s+(ant(erior)?\s)?(vit(reous)?)\s?([a-z]*\s){1,10})'
feregex=r'(?i)((optic\snerve)|(optic\s)?(\sd(is(c|k))?\s).*\s+((c(\/)?d(r)?\s)|((cup(\s|-)(to(\s|-))?dis(c|k)(\s|-))?ratio\s)).*\s+(m(ac(ula)?)?\s).*\s+(v(es(s)?(els)?)?\s).*\s+(p(eriph(ery)?)?\s)([a-z]*\s){1,20})'
sleferegex=r'(?i)(((Slit Lamp )|(SLE )).*\s+(((eye)?l(ids)?(\/|&|(\sand\s))?l(ashes)?)).*\s+((c(onjunctiva)?(\/|&|(\sand\s))?s(clera)?)).*\s+(k|(cornea)).*\s+a(nterior)?\s?c(hamber)?.*\s+(iris).*\s+(lens).*\s+(ant(erior)?\s)?(vit(reous)?)\s?.*\s+)(((post(erior)?\s)?(vit(reous)?)\s.*\s)?(optic\snerve)|(optic\s)?(d(is(c|k))?\s).*\s+((c(\/)?d(r)?\s)|((cup(\s|-)(to(\s|-))?dis(c|k)(\s|-))?ratio\s))?.*\s+(m(ac(ula)?)?\s)?.*\s+(v(es(s)?(els)?)?\s)?.*\s+(p(eriph(ery)?)?\s)?([a-z]*\s){1,10})?'

slellregex=r'(?i)(\s(eye)?l(ids)?(\/|&|(\sand\s))?l(ashes)?)'
slecsregex=r'(?i)(\sc(onjunctiva)?(\/|&|(\sand\s))?s(clera)?)'
slekregex=r'(?i)(\sk|(cornea))'
sleacregex=r'(?i)(\sa(nterior)?\s?c(hamber)?)'
sleirisregex=r'(?i)(iris)'
slelensregex=r'(?i)(lens)'
slevitregex=r'(?i)(ant(erior)?\s)?(vit(reous)?)(\s|:)?'#(((post(erior)?\s)?(vit(reous)?)'
fediscregex=r'(?i)(optic\snerve)|(optic\s)?(dis(c|k)?(\s|:))'
fecdrregex=r'(?i)((c(\/)?d(r)?(\s|:))|((cup(\s|-)(to(\s|-))?dis(c|k)(\s|-))?ratio(\s|:)))'
femacregex=r'(?i)(\sm(ac(ula)?)?(\s|:))'
fevessregex=r'(?i)(\sv(es(s)?(els)?)?(\s|:))'
feperiphregex=r'(?i)(\sp(eriph(ery)?)?(\s|:))'
sleferegexes=[slellregex, slecsregex, slekregex, sleacregex, sleirisregex, slelensregex, slevitregex,fediscregex, fecdrregex, femacregex, fevessregex, feperiphregex]

labeldict = {'ll': ('sleodll', 'sleosll'),
 'cs': ('sleodcs', 'sleoscs'),
 'k': ('sleodk', 'sleosk'),
 'ac': ('sleodac', 'sleosac'),
 'iris': ('sleodiris', 'sleosiris'),
 'lens': ('sleodlens', 'sleoslens'),
 'vit': ('sleodvit', 'sleosvit'),
 'disc': ('feoddisc', 'feosdisc'),
 'cdr': ('feodcdr', 'feoscdr'),
 'mac': ('feodmac', 'feosmac'),
 'vess': ('feodvess', 'feosvess'),
 'periph': ('feodperiph', 'feosperiph')}

#convert span to token index helper function 
def find_token_ind(tokenspanlist, spanstart, spanend): 
    indexmatch=[]
    for i in range(len(tokenspanlist)): 
        span = tokenspanlist[i]
        if span[0]>=spanstart and span[1]<=spanend: 
            indexmatch.append(i)
    return indexmatch
    
def find_row_finding_spans(note, full_regex_match_ind): 
    title=['ll', 'cs', 'k', 'ac', 'iris', 'lens', 'vit', 'disc', 'cdr', 'mac', 'vess', 'periph']
    headerdict = {}
    findingsdict={}
    currindex=full_regex_match_ind
    for regex, title in zip(sleferegexes,title): 
        #print("searching for ", title)
        #print("current index is", currindex)
        #print(note[currindex:])
        match = re.search(regex,note[currindex:]) 
        if match: 
            start=match.start()+currindex
            end=match.end()+currindex
            currindex=end
            #print("new current index is ", currindex)
            #print((start, end), note[start:end])
            if title in headerdict:
                    pass 
            else: 
                headerdict[title]=(start, end, match.group(0))
        else: 
            if title not in ['vit', 'cdr']: #make the vitreous and cdr optional, as many don't document this, otherwise break if we don't find the next row header 
                break 
    for i in range(len(headerdict.values())): 
        key=list(headerdict.keys())[i]
        span=list(headerdict.values())[i]
        #print(key, span)
        try: nextspan=list(headerdict.values())[i+1]
        except IndexError: 
            nextspan=(span[1]+30, span[1]+100) 
        interimspan=(span[1], nextspan[0], span[2])
        if key in findingsdict: 
            pass 
        else: 
            findingsdict[key]=interimspan
            #print(note[interimspan[0]: interimspan[1]])
    return findingsdict 
    
def splitodosfindings(note, tokenspanlist, findingsdict): 
    splitfindingsdict={}
    for key, value in zip(findingsdict.keys(), findingsdict.values()): 
        #print(key, note[value[0]:value[1]])
        #print("looking for left eye header matching ", value[2])
        match = re.search(value[2],  note[value[0]:value[1]])
        if match: #if there's a match found (i.e., a left eye header), then split right and left findings accordingly 
            odspan = (value[0], match.start()+value[0])
            osspan = (match.end()+value[0], value[1])
            odindices = find_token_ind(tokenspanlist, odspan[0], odspan[1])
            osindices = find_token_ind(tokenspanlist, osspan[0], osspan[1])
            #print(odindices, osindices)
            odkey=labeldict[key][0]
            oskey=labeldict[key][1]
            splitfindingsdict[odkey]=odindices
            splitfindingsdict[oskey]=osindices
        else: #if no match found (i.e., no left eye header), then split down the middle by token 
            ouindices = find_token_ind(tokenspanlist, value[0], value[1])
            #print(ouindices)
            odindices = ouindices[:len(ouindices)//2]
            osindices = ouindices[len(ouindices)//2:]
            #print(odindices,osindices)
            odkey=labeldict[key][0]
            oskey=labeldict[key][1]
            splitfindingsdict[odkey]=odindices
            splitfindingsdict[oskey]=osindices
    return splitfindingsdict
    
def return_regex_labels(splitfindingsdict, tokenlistlabels): 
    for name, match in zip(splitfindingsdict.keys(), splitfindingsdict.values()): 
        if len(match)>0: 
            matchstart=match[0]
            matchend=match[-1]
            if tokenlistlabels[matchstart]=='O': #if first available match is "free" or unassigned
                tokenlistlabels[matchstart]='B-'+name #label entity start 
                if matchend>matchstart: #for multiple token matches, label entity continuation 
                    for i in range(matchstart+1, matchend+1): 
                        tokenlistlabels[i]='I-'+name
    return tokenlistlabels 
    
def getregextokenlabels(labelset, inputfilepath, outputfilepath, samplesize, startat, header, regexone, regextwo): 
    '''
    labelset is going to be slefe (the usual) or slefe_nolabel_test (for the nolabel testset) 
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
    
    elif labelset == 'slefe_nolabel_test':
        from slefelabelnames import labelnames
        df=pd.read_csv(inputfilepath)
        df.columns = df.columns.str.strip().str.lower()
        df = df.sample(n=300, random_state = 0) 
    
    print(df.dtypes)
    print(df.head())
    df["note"]=df["note"].apply(remove_unicode_specials)
    logger.debug("length of input dataframe:" +str(len(df)))
    
    #keep rows where any labelnames are not null (drop rows with no labels extracted from sql)
    #saves a little time as we don't need to process these rows 
    if (labelset != 'slefe_nolabel_test'):
        df=df[df[labelnames].notnull().any(1)]
    
    print("tokenizing...")
    #tokenize
    t = TreebankWordTokenizer()
    df["tokens"]=df["note"].apply(t.tokenize)
    df["tokenspanlist"]=df["note"].apply(gettokenspanlist)
    df["doclength"]=df["tokenspanlist"].apply(len)
    df["full_regex_match"] = df["note"].apply(lambda x: select_longest_regex_finding(regexone, regextwo, x))
    df["full_regex_match_ind"] = [x[0].index(x[1]) if x[0] is not None and x[1] is not None else -1 \
                                  for x in zip(df['note'], df['full_regex_match'])]#df.apply(lambda x: x.note.index(x.full_regex_match), axis=1)
    #print(df.head())
    #initialize tokenlist labels to all 'O'
    df["tokenlistlabels"]=df["tokenspanlist"].apply(initializetokenlistlabels)
    
    #returns spans of the findings for each row 
    df["findingsdict"]=df[["note", "full_regex_match_ind"]].apply(lambda x: find_row_finding_spans(*x), axis=1)
    #split the findings into right and left, and convert to token indices 
    df["splitfindingsdict"]=df[["note", "tokenspanlist", "findingsdict"]].apply(lambda x: splitodosfindings(*x), axis=1)
    #update the tokenlistlabels 
    df["tokenlistlabels"]=df[["splitfindingsdict","tokenlistlabels"]].apply(lambda x: return_regex_labels(*x), axis=1)
    
    #print(df.head())
    
    logger.debug("length of resulting dataframe: "+ str(len(df)))
    
    logger.debug("saving output dataframe to "+ outputfilepath)
    df[["note_deid", "doclength", "tokens", "tokenlistlabels"]].to_csv(outputfilepath, index=False)
    
    end = time.time()
    logger.debug("elapsed time: "+ str(end - start)+ " seconds")
    return 
    

labelset='slefe'
#test files
#inputfilepath = 'C:/Users/Sophia/Documents/ResearchPHI/STRIDE_FULL/oph-notes-ner/data/slefe/slefelabels.csv'
#outputfilepath = 'C:/Users/Sophia/Documents/ResearchPHI/STRIDE_FULL/oph-notes-ner/data/slefe/sleferegexlabels-sample.csv'

#test set 
#inputfilepath = 'C:/Users/Sophia/Documents/ResearchPHI/STRIDE_FULL/oph-notes-ner/data/slefe/test_regex.csv'
#outputfilepath = 'C:/Users/Sophia/Documents/ResearchPHI/STRIDE_FULL/oph-notes-ner/data/slefe/sleferegexlabels-test-sophia.csv'

#outset 
labelset='slefe_nolabel_test'
inputfilepath = "C:/Users/Sophia/Documents/ResearchPHI/STRIDE_FULL/oph-notes-ner/data/slefe/slefenolabeltestset-filtered.csv"
outputfilepath = 'C:/Users/Sophia/Documents/ResearchPHI/STRIDE_FULL/oph-notes-ner/data/slefe/sleferegexlabels-outsetsample300-sophia.csv'

samplesize=300
skiprows=0
header=0
logger=MyLogger("logs/regex_sophia_outset-300.log")

getregextokenlabels(labelset, inputfilepath, outputfilepath, samplesize, skiprows, header, sleferegex, sleregex)
