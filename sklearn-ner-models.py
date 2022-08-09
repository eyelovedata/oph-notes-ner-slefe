import pandas as pd
import numpy as np
from seqeval.metrics import f1_score, accuracy_score, classification_report 
import pickle
import ast
import sys
from slefelabelnames import id2tag, tag2id 
import os
from google.cloud import storage
import pickle
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn import ensemble, tree

def get_emr_embedding_dict(bucket_name, path_to_embeddings_folder):
    """
    Get EMR embedding dictionary from GCP storage bucket
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=path_to_embeddings_folder, delimiter='/')
    for blob in blobs:
        if(blob.name != path_to_embeddings_folder):
            file_name = blob.name.replace(path_to_embeddings_folder, "")
            blob.download_to_filename(file_name)
            if (file_name == "emrdict.pickle"):
                emrdict = pickle.load(open(file_name, "rb"))
    return emrdict

def get_train_val_test_data(bucket_name, path_to_data_folder):
    """
    Get train, validation, and testing data from GCP storage bucket
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    data_blobs = bucket.list_blobs(prefix=path_to_data_folder, delimiter='/')
    for blob in data_blobs:
        if(blob.name != path_to_data_folder):
            file_name = blob.name.replace(path_to_data_folder, "")
            if (file_name == "train.csv"):
                blob.download_to_filename(file_name)
                train_df = pd.read_csv(file_name, header=0)
            elif (file_name == "val.csv"):
                blob.download_to_filename(file_name)
                val_df = pd.read_csv(file_name, header=0)
            elif (file_name == "test.csv"):
                blob.download_to_filename(file_name)
                test_df = pd.read_csv(file_name, header=0)

    # Remove unneded data columns
    del train_df["doclength"]; del train_df["labelcount"];
    del train_df["detectedlabelcount"]; del train_df["ratiodetectedlabels"]
    del val_df["doclength"]; del val_df["labelcount"];
    del val_df["detectedlabelcount"]; del val_df["ratiodetectedlabels"]
    train_sample_size = len(train_df.index)
    val_sample_size = len(val_df.index)
    return pd.concat([train_df, val_df]),  test_df, \
        train_sample_size, val_sample_size

def get_token_lists(train_val_df, test_df):
    """
    Convert datasets to lists of tokens
    """
    tokenlist_train_val=train_val_df["tokens"].tolist()
    tokenlistlabels_train_val=train_val_df["tokenlistlabels"].tolist()

    tokenlist_test=test_df["tokens"].tolist()
    tokenlistlabels_test=test_df["tokenlistlabels"].tolist()
    return tokenlist_train_val, tokenlistlabels_train_val, tokenlist_test, tokenlistlabels_test

def set_up_train_and_validation_folds(train_sample_size, val_sample_size):
    """
    Set up a defined train and validation split indices to use with sklearn GridSearchCV
    """
    train_indices = np.full(train_sample_size, -1, dtype=int)
    val_indices = np.full(val_sample_size, 0, dtype=int)
    test_fold = np.append(train_indices, val_indices)
    ps = PredefinedSplit(test_fold)
    return ps

def convert_document_list_to_word_vectors(data, embedding_dict):
    #turn the document tokens into an array of word vectors 
    wordvectorarraylist=[]
    datalen = len(data)
    for i, doc in enumerate(data): 
        if (i % 1000 == 0):
            print(str(i + 1), "/" + str(datalen))
        wordlist=ast.literal_eval(doc)
        wordvectorlist=[]
        for w in wordlist:
            if (str.lower(w) in embedding_dict): 
                wordvectorlist.append(embedding_dict[str.lower(w)])#g.emb(str.lower(w))#[str.lower(w)])
            else:
                wordvectorlist.append(embedding_dict["unk"])
        wordvectorarraylist.append(np.array(wordvectorlist))
    X=np.vstack(wordvectorarraylist)#.astype('float16')
    return X

def convert_labels_to_label_list(data):
    #turn the document labels into the Y vector 
    Yarraylist=[]
    datalen = len(data)
    for i, doc in enumerate(data): 
        if (i % 1000 == 0):
            print(str(i + 1) + "/" + str(datalen))
        labellist=ast.literal_eval(doc)
        labelidlist=[]
        for l in labellist: 
            labelidlist.append(tag2id[l])
        Yarraylist.append(np.array(labelidlist))
    Y=np.hstack(Yarraylist).astype('int8')
    return Y

def set_random_forest_hyperparameter_grid(Y_train_weights):
    """
    Set hyperparameters to search for optimizing random forest model
    """
    n_estimators = [50, 100]
    # Number of features to consider at every split
    max_features = ['sqrt', 'log2']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 30, num = 3)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [5, 10, 15]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [2, 4, 6]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the hyperparamter grid
    hyperparameter_grid = {'n_estimators': n_estimators,
                           'max_features': max_features,
                           'max_depth': max_depth,
                           'min_samples_split': min_samples_split,
                           'min_samples_leaf': min_samples_leaf,
                           'bootstrap': bootstrap,
                           'class_weight': ["balanced"]}
    return hyperparameter_grid

def set_decision_tree_hyperparameter_grid(Y_train_weights):
    """
    Set hyperparameters to search for optimizing decision tree model
    """
    # Max tree depth
    max_depth = [int(x) for x in np.linspace(10, 30, num=3)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [15, 10, 5]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [2, 4, 6]
    # Optimization criteria
    criterion = ['entropy', 'gini']
    hyperparameter_grid = {'max_depth': max_depth,
                           'min_samples_split': min_samples_split,
                           'min_samples_leaf': min_samples_leaf,
                           'criterion': criterion,
                           'class_weight': ["balanced"]}
    return hyperparameter_grid

def run_hyperparameter_grid_search(hparam_grid, model_type, X_train_val, Y_train_val, ps):
    """
    Run hyperparameter grid search with given hyperparameters and model type
    """
    if (model_type == "random_forest"):
        clf = GridSearchCV(estimator=ensemble.RandomForestClassifier(),
                           param_grid=hparam_grid,
                           cv=ps)
    elif (model_type == "decision_tree"):
        clf = GridSearchCV(estimator=tree.DecisionTreeClassifier(),
                           param_grid=hparam_grid,
                           cv=ps)
    else:
        raise(ValueError("Model type can only be \"random_forest\" or " \
                         + "\"decision_tree\""))
    clf.fit(X_train_val, Y_train_val)
    print(model_type)
    print(clf.best_params_)
    return clf.best_params_
    
def evaluate_model(clf, tokenlist_test, tokenlistlabels_test, model_name, emrdict):
    print("Evaluating " + model_name + "...")
    # Getting predictions
    wordvectorarraylist = []
    for doc in tokenlist_test: 
        wordlist=ast.literal_eval(doc)
        wordvectorlist=[]
        for w in wordlist:
            if (str.lower(w) in emrdict): 
                wordvectorlist.append(emrdict[str.lower(w)])
            else:
                wordvectorlist.append(emrdict["unk"])
        wordvectorarraylist.append(np.array(wordvectorlist))
    Y_pred_test = []
    for doc in wordvectorarraylist: 
        Y_pred_doc = clf.predict(doc)
        Y_pred_test.append(list(Y_pred_doc))
    pred_tags = [[id2tag[label] for label in doc] for doc in Y_pred_test]

    # Getting labels
    Yarraylist=[]
    for doc in tokenlistlabels_test: 
        labellist=ast.literal_eval(doc)
        labelidlist=[]
        for l in labellist: 
            labelidlist.append(tag2id[l])
        Yarraylist.append(labelidlist)
    valid_tags = [[id2tag[label] for label in doc] for doc in Yarraylist]
    print(classification_report(valid_tags, pred_tags))


def main(): 
    bucket_name = "stanfordoptimagroup"
    path_to_embeddings_folder = "STRIDE/neuralwordembeddings/"
    path_to_data_folder = "STRIDE/oph-notes-ner/data/slefe/"
    print("Getting EMR embedding dict...")
    emrdict = get_emr_embedding_dict(bucket_name, path_to_embeddings_folder)
    print("Getting train, val, and test sets...")
    train_val_df, test_df, train_sample_size, val_sample_size = get_train_val_test_data(bucket_name, path_to_data_folder)
    print("Converting train, val, and test sets to token lists...")
    tokenlist_train_val, tokenlistlabels_train_val,tokenlist_test, tokenlistlabels_test = get_token_lists(train_val_df, test_df)

    # Setting up train, val, and test set variables
    print("Setting up lists of word vectors...")
    X_train_val = convert_document_list_to_word_vectors(tokenlist_train_val, emrdict)
    #X_test = convert_document_list_to_word_vectors(tokenlist_test, emrdict)
    Y_train_val = convert_labels_to_label_list(tokenlistlabels_train_val).astype('int8')
    #Y_test = convert_labels_to_label_list(tokenlistlabels_test)
    ps = set_up_train_and_validation_folds(train_sample_size, val_sample_size)

    # Setting up weighting dictionary to account for largely imbalanced data
    print("Computing weighting dictionary...")
    uniques, counts = np.unique(Y_train_val, return_counts=True)
    inverse_freqs = 1/counts
    Y_train_weights = {a:b for (a, b) in zip(uniques.astype('int8'), inverse_freqs)}

    # Set up hyperparameters
    print("Setting hyperparameter search grid...")
    rf_hparam_grid = set_random_forest_hyperparameter_grid(Y_train_weights)
    dt_hparam_grid = set_decision_tree_hyperparameter_grid(Y_train_weights)

    # Run hyperparameter grid searches
    print(np.unique(Y_train_val))
    print(np.unique(Y_train_val[train_sample_size:]))
    print("Running hyperparameter search...")
    best_params_rf = run_hyperparameter_grid_search(rf_hparam_grid, "random_forest",
                                   X_train_val, Y_train_val, ps)
    best_params_dt = run_hyperparameter_grid_search(dt_hparam_grid, "decision_tree",
                                   X_train_val, Y_train_val, ps)

    # Manually inputting the best hyperparams that were found in the previous hyperparam search
    """
    best_params_rf = {'bootstrap': True, 'max_depth': 20, 'max_features': 'sqrt',
                      'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 50,
                      'class_weight': Y_train_weights}
    best_params_dt = {'criterion': 'entropy', 'max_depth': 10, 'min_samples_leaf': 6,
                      'min_samples_split': 15, 'class_weight': Y_train_weights}
    """
    print("Best random forest params")
    print(best_params_rf)
    print("Best decision tree params")
    print(best_params_dt)

    clf_rf_opt = ensemble.RandomForestClassifier(n_estimators=best_params_rf['n_estimators'],
                                                 max_features=best_params_rf['max_features'],
                                                 bootstrap=best_params_rf['bootstrap'],
                                                 max_depth=best_params_rf['max_depth'],
                                                 min_samples_leaf=best_params_rf['min_samples_leaf'],
                                                 min_samples_split=best_params_rf['min_samples_split'],
                                                 class_weight=best_params_rf['class_weight'])
    clf_dt_opt = tree.DecisionTreeClassifier(criterion=best_params_dt['criterion'],
                                             max_depth=best_params_dt['max_depth'],
                                             min_samples_leaf=best_params_dt['min_samples_leaf'],
                                             min_samples_split=best_params_dt['min_samples_split'],
                                             class_weight=best_params_dt['class_weight'])
    print("Fitting random forest...")
    clf_rf_opt.fit(X_train_val, Y_train_val)
    print("Fitting decision tree...")
    clf_dt_opt.fit(X_train_val, Y_train_val)
    evaluate_model(clf_rf_opt, tokenlist_test, tokenlistlabels_test, "random_forest", emrdict)
    evaluate_model(clf_dt_opt, tokenlist_test, tokenlistlabels_test, "decision_tree", emrdict)

if __name__ == "__main__":
    main()
