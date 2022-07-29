labelnames=['sleodll', 'sleosll', 'sleodcs', 'sleoscs', 'sleodk', 'sleosk', 'sleodac', 'sleosac', 'sleodiris', 'sleosiris', 'sleodlens', 'sleoslens', 'sleodvit', 'sleosvit', 'feoddisc','feosdisc','feodcdr','feoscdr','feodmac','feosmac','feodvess','feosvess','feodperiph','feosperiph']

keynameslist=[]
for name in labelnames: 
	keynameslist.extend(['B-'+name, 'I-'+name])
tag2id={tag: id for id, tag in enumerate(keynameslist, start=1)}
tag2id['O']=0
id2tag = {id: tag for tag, id in tag2id.items()}
inferenceid2tag={"LABEL_"+str(id): tag[2:] for tag, id in tag2id.items()}
