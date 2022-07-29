# prodigy custom for va labels 

import prodigy
from prodigy.components.loaders import JSONL
from prodigy.components.preprocess import add_tokens
import spacy


@prodigy.recipe("custom-ner")
def custom_ner_recipe(dataset, source):
    stream = JSONL(source)                          # load the data
    #stream = add_entities_to_stream(stream)         # add custom entities
    stream = add_tokens(spacy.blank("en"), stream, skip=True)  # add "tokens" to stream

    return {
        "dataset": dataset,          # dataset to save annotations to
        "stream": stream,            # the incoming stream of examples
        "view_id": "ner_manual",     # annotation interface to use
		"config": {
            "validate": False, 
			"labels": ['sleodll', 'sleosll', 'sleodcs', 'sleoscs', 'sleodk', 'sleosk', 'sleodac', 'sleosac', 'sleodiris', 'sleosiris', 
			'sleodlens', 'sleoslens', 'sleodvit', 'sleosvit', 'feoddisc','feosdisc','feodcdr','feoscdr','feodmac','feosmac','feodvess','feosvess','feodperiph','feosperiph'],  # labels to annotate
}
    }
	