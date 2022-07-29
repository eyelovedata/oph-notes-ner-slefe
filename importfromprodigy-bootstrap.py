import json
from seqeval.metrics import accuracy_score, classification_report, f1_score
import spacy
from spacy.gold import biluo_tags_from_offsets
import random 
import pandas as pd 
import numpy as np 
from utils import MyLogger

'''
This script imports two jsonl files from prodigy, converts to format that works with seqeval, 
and runs the classification report comparing the two. 
it is labelset agnostic - presuming the two files improted from prodigy have the same labelset, it doesn't matter what they are

This version includes bootstrapping confidence intervals
'''
n_replicates = 1000 #bootstrapping replicates
iter_print = 10 #print counter for bootstrapping replicates 
output_results_path = "prodigyfiles/slefe/outset/output/regex-outset-performance-ci.csv"

logger=MyLogger("logs/importfromprodigy-bootstrap-regex-outset-predictionsvscorrected-revision.log") 

def jsontobiotags(jsonlfilepath, startline=0): 
    result=[]
    with open(jsonlfilepath) as f:
        for line in f: 
            result.append(json.loads(line))
            
    listoftags=[]
    nlp = spacy.blank("en")
    for item in result: 
        text=item['text']
        spans=item['spans']
        doc = nlp(text)
        offsets = [(span["start"], span["end"], span["label"]) for span in spans]
        biluo_tags = biluo_tags_from_offsets(doc, offsets)
        doc.ents = [doc.char_span(start, end, label) for start, end, label in offsets]
        iob_tags = [f"{t.ent_iob_}-{t.ent_type_}" if t.ent_iob_ else "O" for t in doc]
        listoftags.append(iob_tags)
    return listoftags[startline:]

startline=0
correctedtagpath="prodigyfiles/slefe/outset/output/regex-outset-predictions-corrected-300.jsonl"
correctedtags=jsontobiotags(correctedtagpath, startline=startline)
logger.debug("corrected tags come from file"+correctedtagpath)

modeltagpath="prodigyfiles/slefe/outset/output/regex-outset-predictions-orig-300.jsonl"
modeltags=jsontobiotags(modeltagpath, startline=0)
logger.debug("original tags come from file"+modeltagpath) 

print("tags in each list:", len(correctedtags), len(modeltags)) 

newmodeltags=[]
newcorrectedtags=[]
for modeltag, correctedtag in zip(modeltags[0:len(correctedtags)], correctedtags): 
	if len(modeltag) != len(correctedtag): 
		#print(len(modeltag), len(correctedtag))
		pass 

	else: 
		newmodeltags.append(modeltag)
		newcorrectedtags.append(correctedtag)

print("tags remaining in each list:", len(newmodeltags), len(newcorrectedtags)) 

logger.debug("Validation Accuracy: " + str(accuracy_score(correctedtags, modeltags[0:len(correctedtags)])))
#logger.debug("F1 Score: " + str(f1_score(correctedtags, modeltags[0:len(correctedtags)])))

logger.debug("Calculating Main Results (Original Scores):") 
#logger.debug(classification_report(newcorrectedtags, newmodeltags))
dfoutput=pd.DataFrame(classification_report(newcorrectedtags, newmodeltags, output_dict=True)).transpose() 
print(dfoutput) 

#logger.debug("calculating confidence intervals over ", n_replicates, " replicates...") 

#create list of dataframes 
replicate_results = [] 

#sample from predictions 
#this is a really slow script because running classification_report is really slow! 
for i in range(n_replicates): 
	if i%iter_print==0: 
		print(i) 
	sample_corrected, sample_model = zip(*random.choices(list(zip(newcorrectedtags, newmodeltags)), k=len(newmodeltags)))
	output=classification_report(sample_corrected, sample_model, output_dict=True)
	#store pandas dataframe as list 
	replicate_results.append(np.expand_dims(np.array(pd.DataFrame(output).transpose()[["precision", "recall", "f1-score"]]), axis=2)) 

npreplicates = np.concatenate(replicate_results, axis=2) 
#get quantile along axis 
lower_bound = np.quantile(npreplicates, 0.025, axis=2) 	
upper_bound = np.quantile(npreplicates, 0.975, axis=2)


df_lower=pd.DataFrame(lower_bound, columns=['precision_lower', 'recall_lower', 'f1_lower'], index=dfoutput.index) 
df_upper=pd.DataFrame(upper_bound, columns=['precision_upper', 'recall_upper', 'f1_upper'], index=dfoutput.index) 

finalresultdf = pd.concat([dfoutput["precision"], df_lower["precision_lower"], df_upper["precision_upper"], 
dfoutput["recall"], df_lower["recall_lower"], df_upper["recall_upper"], 
dfoutput["f1-score"], df_lower["f1_lower"], df_upper["f1_upper"], 
dfoutput["support"]], axis=1) 

print(finalresultdf) 

print("saving results to ", output_results_path) 
finalresultdf.to_csv(output_results_path)
