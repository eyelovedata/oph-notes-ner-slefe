# Named Entity Recognition for Ophthalmology Clinical Notes To Recognize Slit Lamp Exam and Fundus Exam 

# Requirements 
Please use Anaconda to manage environments and use the included environment.yml file to recreate the environment. 
Notably, however, these scripts make use of the huggingface transformers library and train using PyTorch. 

# Purpose 
- to recognize the eye exam compponents of free-text ophthalmology progress notes from Stanford electronic health records
- We try the following sets of entities to recognize SLE and FE exam entities
- SLE and FE entities are the standard headers for those components of the eye exam, e.g. lids/lashes, conj/sclera, d/m/v/p etc. 

# General Approach 
- We use use pre-trained distilbert, biobert, clinicalbert, and bluebert as base models which we finetune to our specific purpose. 
- We utilize the huggingface transformers library.  

# Scripts 
## Preprocessing Data 
- extract-labeled-notes.sql: A record of the sql queries developed to extract the visual acuity and slit lamp exam data and the corresponding notes from our database of EHR information. 
- tokenizeandlabel.py: Takes the output of the above sql query and performs preprocessing to propagate document-level labels to token-level labels. To do this, this script tokenizes the notes, does text search for the documented eye exam over those tokens, and assigns tokens to labels using a heuristic. The outputs are token-level labels saved to a csv file that is one row per note. This script only needs to be run once on the raw data. 
- traintestsplitbynote.py: Performs a train/val/test split, splitting at the document (note) level. 
- prepareforbert.py: Script which prepares the data for input into PyTorch models, including splitting long documents into shorter subdocuments, and performing WordPieceTokenization. Each model has different input length requirements and different tokenization requirements, so this script takes a parameter which specifies which model type you want to preprocess for. Must be run separately for each different model. Outputs are preprocessed datasets stored as custom PyTorch dataset classes with extention ".pt" which are ready to load into the model using torch.load() 

## Model Training and Evaluation Statistics 
- trainmodel.py: Trains the desired BERT model, using the the desired datasets, outputting the saved model to desired location. 
- finetunemodel.py: Script used for hyperparameter tuning BERT models 
- evalmodel-bootstrap.py: Loads a saved model, runs predictions over a validation set of your choosing, and computes performance metrics of validation loss, accuracy, F1-score, and classification report per class usin seqeval package. Includes bootstrapping to desired number of replicates to produced 95% confidence intervals. 
- regex_label.py: script for running the baseline model on either the test set or the outset to produce predictions 

## Model Error Analysis with Prodigy 
The purpose of this set of scripts is to be able to use a user-friendly interface for annotating notes (Prodigy, https://prodi.gy/), to visualize, evaluate, and correct our models' output predictions. The goal is to take a sample of the model's predictions on the validation set, load into Prodigy, view which words are highlighted as entities, and correct these predictions. Then we can evaluate how close the model's predictions are to the human-level "ground truth". This extra step is important because the model isn't trained on human-level ground truth labels, but rather on this weakly-supervised labels as described above in preprocessing steps. 
- exporttoprodigy.py: Takes a set of validation notes and labels, performs inference on them using BERT model of choice, and produces .jsonl files of the predicted labels and original labels, which can be loaded into Prodigy for further annotation. 
- exporttoprodigyregexonly.py: Takes saved baseline model predictions and changes formatting so they can be loaded into Prodigy for further annotation. 
- importfromprodigy-bootstrap.py: Prodigy saves annotations into .jsonl files. This script convers the .jsonl files back into a format which can be input into seqeval and can run classification metric on them. Includes bootstrapping to desired number of replicates to produce 95% confidence intervals. 
- prodigycustomslefe.py: Custom recipe for Prodigy to display combined SLE/FE annotations correctly
- sample-export-ouset.py: Takes a set of notes with no labels, comprising a group of notes written without the benefit of SmartForm templates, samples 300 of them, runs inference with the model of choice, and exports to Prodigy-friendly format for human annotation. 

## Utilities 
- utils.py: A collection of utility functions called by other functions in above scripts. Also contains definitions for OphNERDataset class (PyTorch dataset) 
- slefelabelnames.py: a variety of dictionaries of labels, label ids, etc. specific to identifying slit lamp exam and fundus exam entities 
