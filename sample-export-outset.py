import torch 
import numpy as np
from utils import OphNERDataset, MyLogger, remove_unicode_specials, singledocinference, multipledocinference   
from transformers import DistilBertForTokenClassification, pipeline
import json
import random 
import pandas as pd 

logger=MyLogger("logs/sample-export-outset-bluebert-revision.log")


'''task: read in notes that do not have smartform labels, sample them, 
preprocess the sample for bert (but do so without labels!),
do predictions/inference over them, and then export to prodigy to get ready for labeling''' 

#choose labelset 
labelset = "slefe" 
if labelset == "slefe": 
    from slefelabelnames import labelnames, tag2id, id2tag, inferenceid2tag

#params 
modeltype = "bluebert_pm"
inputmodelpath='../models/'+labelset+'/'+modeltype+'_finetuned'
val_dataset_path='../data/'+labelset+'/slefenolabeltestset-filtered.csv'
outputjsonpredictionspath="prodigyfiles/"+labelset+"/outset/"+modeltype+"-predictions.jsonl"

#perform inference over data sample 

def splitstring(stringlist, size): 
    '''Makes a list of long strings into a list of shorter strings. 
    Size indicates desired length (number of characters) of the shorter strings'''
    
    shortstringlist=[]
    
    #iterate through each document 
    for i in range(len(stringlist)): 
        doc=stringlist[i]
        #split up each doc into a list of lists of tokens and tags 
        stringlistbroken=[doc[x:x+size] for x in range(0, len(doc), size)]
        #tack on the lists of lists into the master lists of lists
        shortstringlist.extend(stringlistbroken)
    return shortstringlist

def predictions_to_prodigy(modeltype, inputmodelpath, val_dataset_path, outputjsonpredictionspath): 
    '''
    - performs predictions using loaded model on specified dataset
    - save the predictions to a jsonl file which can then be loaded into prodigy for manual annotation/correction 
    '''
    #load our trained model with its associated tokenizer into the ner pipeline
    #create a classifier 
    
    logger.debug('loading model from '+ inputmodelpath)
    from transformers import pipeline
    
    #load model 
    if modeltype=="distilbert": 
        from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast
        model = DistilBertForTokenClassification.from_pretrained(inputmodelpath, local_files_only=True)
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased', model_max_len=512)

    if modeltype=="clinicalbert": 
        from transformers import BertForTokenClassification, BertTokenizerFast 
        model = BertForTokenClassification.from_pretrained(inputmodelpath, local_files_only=True)
        tokenizer = BertTokenizerFast.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", model_max_len=128)
        
    if modeltype =="biobert": 
        from transformers import BertForTokenClassification, BertTokenizerFast
        model = BertForTokenClassification.from_pretrained(inputmodelpath, local_files_only=True)
        tokenizer = BertTokenizerFast.from_pretrained("dmis-lab/biobert-v1.1", model_max_len=512)
        
    if modeltype=="bluebert_pm":
        from transformers import BertForTokenClassification, BertTokenizerFast
        model = BertForTokenClassification.from_pretrained(inputmodelpath, local_files_only=True)
        tokenizer = BertTokenizerFast.from_pretrained('bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12', model_max_len=128)    

    classifier = pipeline('ner', model=model, tokenizer=tokenizer, grouped_entities=False, ignore_labels=['LABEL_0'])
    
    #read in data and sample it 
    logger.debug("reading data in from" + val_dataset_path) 
    
    df = pd.read_csv(val_dataset_path, names=['note_deid', 'provider_deid', 'pat_deid', 'note'], low_memory=True) 
    df = df.sample(n=300, random_state = 0) 
    df.to_csv('../data/'+labelset+'/slefenolabeltestset-filtered-sampled.csv')
    
    #data preprocessing
    df["note"]=df["note"].apply(remove_unicode_specials)

    doclist = df["note"].tolist() 
    
    shortdoclist = splitstring(doclist, 750) 
    logger.debug("Total number of subdocuments: "+str(len(shortdoclist))) 

    logger.debug("performing inference") 
    multipledocinference(shortdoclist, classifier, outputjsonpredictionspath, inferenceid2tag)
    

predictions_to_prodigy(modeltype, inputmodelpath, val_dataset_path, outputjsonpredictionspath)
