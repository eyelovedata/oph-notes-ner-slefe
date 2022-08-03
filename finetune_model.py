import sys
import pandas as pd
import re 
import nltk 
import numpy as np
import ast
import string 
import math
import torch
from tqdm.notebook import tqdm
from utils import OphNERDataset, MyLogger 
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback 
from seqeval.metrics import f1_score, accuracy_score, classification_report
import optuna

#parameters
#specify which model we want to train for. One of three choices ("distilbert", "biobert", "clinicalbert", "bluebert_p", "bluebert_pm") 
#modeltype = "distilbert"
#modeltype = "biobert"
#modeltype = "clinicalbert"
3modeltype="bluebert_p"
modeltype="bluebert_pm"

#specify which labelset we want to train for 
labelset="slefe" 
#file_suffix = '-10000'
outputmodelpath='../models/'+labelset+'/'+modeltype+'.model'
#inputmodelpath='../models/'+labelset+'/'+modeltype+'-sample10000'
inputmodelpath = False # performing fine tuning of base models
train_dataset_path='../data/'+labelset+'/preprocessed/train-'+modeltype+'.pt'
val_dataset_path='../data/'+labelset+'/preprocessed/val-'+modeltype+'.pt'

#specify number of trials to fine tune the model
n_trial = 5

logger=MyLogger("logs/trainmodel-"+labelset+"-"+modeltype+".log")

#logger.debug(torch.cuda.get_device_name(0))

def trainmodel(labelset, modeltype, train_dataset_path, val_dataset_path, outputmodelpath, inputmodelpath=None, n_trials): 
    logger.debug("training model type: "+modeltype)
    logger.debug("labels we are training for: "+labelset) 
    if labelset == "va": 
        from valabelnames import labelnames, tag2id, id2tag 
    if labelset == "sle": 
        from slelabelnames import labelnames, tag2id, id2tag 
    if labelset == "fe": 
        from felabelnames import labelnames, tag2id, id2tag 
    if labelset == "slefe" 
        from slefelabelnames import labelnames, tag2id, id2tag    

    print("loading datasets...") 

    logger.debug("train dataset: " + train_dataset_path) 
    logger.debug("val dataset: " + val_dataset_path) 

    train_dataset=torch.load(train_dataset_path)
    val_dataset=torch.load(val_dataset_path)

    logger.debug("length of training dataset: "+str(len(train_dataset.labels)))
    logger.debug("length of validation dataset: "+str(len(val_dataset.labels)))


    torch.cuda.empty_cache()

    training_args = TrainingArguments(
        output_dir='./fine_tune_result_optuna',   # output directory
        eval_steps=500,
        disable_tqdm=True, 
        load_best_model_at_end=True, 
        evaluation_strategy="steps",
        metric_for_best_model = 'eval_loss'
    )

    if modeltype == "distilbert": 
        from transformers import DistilBertForTokenClassification
        if inputmodelpath: 
            logger.debug("loading model from "+inputmodelpath) 
            model = DistilBertForTokenClassification.from_pretrained(inputmodelpath, local_files_only=True)
        else: 
            logger.debug("loading pretrained model...") 
            model = DistilBertForTokenClassification.from_pretrained('distilbert-base-cased', num_labels=len(tag2id))


    elif modeltype=="clinicalbert": 
        from transformers import BertForTokenClassification
        if inputmodelpath: 
            logger.debug("loading model from "+inputmodelpath) 
            model = BertForTokenClassification.from_pretrained(inputmodelpath, local_files_only=True)
        else: 
            logger.debug("loading pretrained model...") 
            model = BertForTokenClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=len(tag2id))

    elif modeltype=="biobert": 
        from transformers import BertForTokenClassification
        if inputmodelpath: 
            logger.debug("loading model from "+inputmodelpath) 
            model = BertForTokenClassification.from_pretrained(inputmodelpath, local_files_only=True)
        else:
            logger.debug("loading pretrained model...") 
            model = BertForTokenClassification.from_pretrained("dmis-lab/biobert-v1.1", num_labels=len(tag2id))
            
    elif modeltype=="bluebert_p": 
        from transformers import BertForTokenClassification
        if inputmodelpath: 
            logger.debug("loading model from "+inputmodelpath) 
            model = BertForTokenClassification.from_pretrained(inputmodelpath, local_files_only=True)
        else:
            logger.debug("loading pretrained model...") 
            model = BertForTokenClassification.from_pretrained("bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12", num_labels=len(tag2id))

    elif modeltype=="bluebert_pm": 
        from transformers import BertForTokenClassification
        if inputmodelpath: 
            logger.debug("loading model from "+inputmodelpath) 
            model = BertForTokenClassification.from_pretrained(inputmodelpath, local_files_only=True)
        else:
            logger.debug("loading pretrained model...") 
            model = BertForTokenClassification.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12", num_labels=len(tag2id))

    def model_init():
        return model

    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.2),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 5, 20),
            #"gradient_accumulation_steps": trial.suggest_int("gradient_accumulation_steps", 2, 5),
            "warmup_steps":trial.suggest_int("warmup_steps", 100, 1000)
            #"seed": trial.suggest_int("seed", 20, 42),
            #"per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16])
        }

    trainer = Trainer(
        model=model_init,                         # model needs to be passed to a model_init function
        args=training_args,                       # training arguments, defined above
        train_dataset=train_dataset,              # training dataset
        eval_dataset=val_dataset,                 # evaluation dataset
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]         #early stopping 
    )

    best_trail = trainer.hyperparameter_search(
           direction="minimize"          # default search direction by searching the minimum validation loss
           hp_space=hp_space,            # pre-defined search space
           backend="optuna",             # use optuna as hyperparameter search backend
           n_trials=n_trials)            # number of trial to search

    #get best hyperparameters value
    logger.debug("Best hyperparameters are: "+ best_trail.hyperparameters) 
    #trainer.save_model(outputmodelpath)
    return best_trail.hyperparameters


trainmodel(labelset, modeltype, train_dataset_path, val_dataset_path, outputmodelpath, n_trials)
