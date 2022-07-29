import numpy as np
from utils import OphNERDataset, MyLogger, defaultlabels
import json
import random 
import pandas as pd 
import ast 

logger=MyLogger("logs/exporttoprodigyregexonly-outset300.log") 

#choose labelset 
labelset = "slefe" 
if labelset == "slefe": 
    from slefelabelnames import labelnames, tag2id, id2tag, inferenceid2tag


#val_dataset_path='../data/'+labelset+'/test.csv'
#the outset
regex_dataset_path='C:/Users/Sophia/Documents/ResearchPHI/STRIDE_FULL/oph-notes-ner/data/slefe/sleferegexlabels-outsetsample300-sophia.csv'
outputregexpath = "prodigyfiles/slefe/outset/regexlabels-outset-300.jsonl" 

#test set
#regex_dataset_path='C:/Users/Sophia/Documents/ResearchPHI/STRIDE_FULL/oph-notes-ner/data/slefe/sleferegexlabels-test-sophia.csv'
#outputregexpath = "prodigyfiles/slefe/regexlabels-300.jsonl" 
samplesize = 300

# read in datasets 

preds_df=pd.read_csv(regex_dataset_path, usecols=["note_deid", "doclength", "tokens", "tokenlistlabels"])
preds_df["doclength"]=pd.to_numeric(preds_df["doclength"], errors='coerce', downcast='integer')
preds_df = preds_df[preds_df["doclength"].notnull()]
preds_df["defaultlabels"]=preds_df["doclength"].apply(defaultlabels)
preds_df.loc[preds_df.tokenlistlabels.isnull(), 'tokenlistlabels'] = preds_df.loc[preds_df.tokenlistlabels.isnull(), 'defaultlabels']

########this section starting here only for the test set - needed because test set isn't sampled already
##not needed for outset export, which is already sampled down to required size 
#dftokenlabels=pd.read_csv(val_dataset_path)
## datasets
#idx=list(set(preds_df["note_deid"]).intersection(set(dftokenlabels["note_deid"])))    
#random.seed(1)
#random.shuffle(idx)

#logger.debug("sampling "+ str(samplesize)+ " notes from the test set")
#random.seed(1) 
#sampleidx=random.sample(idx, samplesize)

##saves the smaller dataset and sorts it into order 
#dftokenlabels = dftokenlabels[dftokenlabels["note_deid"].isin(sampleidx)]
#dftokenlabels.sort_values(by=['note_deid'], inplace=True)
#print('length of tokenlabels', len(dftokenlabels))

##load regex labels and pick out the same sample ids and sort them into same order
#preds_df = preds_df[preds_df["note_deid"].isin(sampleidx)]
#preds_df.sort_values(by=['note_deid'], inplace=True)
##############end test set processing section 

print('length of regexlabels', len(preds_df)) 




def get_entity_token_spans(newdoctokens, newdoclabels): 
    '''
    Outputs lines of json which include the entity span (start and end) 
    for the training labels 
    '''
    entitylist=[] 
    #turn newdoctokens into a list of token spans 
    tokenspans=[(0,len(newdoctokens[0]))]
    for i in range(1,len(newdoctokens)): 
        token=newdoctokens[i]
        start=tokenspans[i-1][1]+1
        end=start+len(token)
        tokenspans.append((start,end))
        label=newdoclabels[i]
        if label!='O':
            entitylist.append({"start":start, "end":end, "label":label})
    jsonline={}
    jsonline["text"]=' '.join([x for x in newdoctokens])
    jsonline["spans"]=entitylist
    return jsonline  
	
def save_labels_to_prodigy(outputfilepath, df): 
    jsonlinelist=[]
    doclist=[]
    for idx in range(len(df)): 
        #get a list of tokens - aligns with the labels 
        wordpiecetokens = ast.literal_eval(df.iloc[idx]["tokens"])
        wordpiecetags = [label.replace('B-','').replace('I-', '') for label in ast.literal_eval(df.iloc[idx]["tokenlistlabels"])]
        #wordpiecetags = [[label for label in ast.literal_eval(doc)] for doc in predictions]
        jsonline = get_entity_token_spans(wordpiecetokens, wordpiecetags)
        jsonlinelist.append(jsonline)
        
        docstring=jsonline["text"] 
        doclist.append(docstring)
    
    print('saving labels file to ', outputfilepath)
    with open(outputfilepath, 'w') as f:
        for jsonline in jsonlinelist:
            f.write(json.dumps(jsonline) + "\n")
    return doclist 
	
	#process regex file
print('process regex file...') 
save_labels_to_prodigy(outputregexpath, preds_df)    

