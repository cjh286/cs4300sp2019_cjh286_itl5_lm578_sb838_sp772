#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
import numpy as np
import copy

# import spacy
from collections import defaultdict 
import csv
import re
import openpyxl
import os
import json

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import sklearn.decomposition
from sklearn.decomposition import TruncatedSVD
import scipy
from scipy import stats
from scipy.sparse.linalg import svds, eigs

# nlp = spacy.load('en_core_web_md')
# # tokens = nlp(u'dog cat banana afskfsd')
# # for token in tokens:
# #     print(token.text, token.has_vector, token.vector_norm, token.is_oov)

def tokenize(text): 
    """ Taken from class code A1
    Returns a list of words that make up the text.
    
    Note: for simplicity, lowercase everything.
    Requirement: Use Regex to satisfy this function
    
    Params: {text: String}
    Returns: List 
    """
    # YOUR CODE HERE
    x = re.findall(r'\b([a-zA-Z]+)\b',text)  #re.findall(r'\b([a-zA-Z]{3,20}+)\b',text)
    r = []
    for y in range(len(x)):
        if len(x[y]) > 2 :
            r.append(x[y].lower())
    return r


def extract_excel_data():
    wiki_descrips_dict = {} #store alc names and review text
    lables_from_csv = {} #since all are labled in csv collect these for testing
    script_path = os.path.abspath(__file__) 
    path_list = script_path.split(os.sep)
    script_directory = path_list[0:len(path_list)-4]
    rel_path = 'scraped_data/ingredient_tastes.csv'
    path = "/".join(script_directory) + "/" + rel_path
    count_undone = 0
    with open(path, encoding="utf8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        first_line_true = 0
        for row in csv_reader:
            if not first_line_true == 0: #this if statement exists to skip the header row in the csv
                if not len(tokenize(row[3])) == 0: #this skips the ingredients with no description
                    wiki_descrips_dict[row[0]] = (row[3]) #tokenize(row[3]) dont want it tokenized since thats done later
                    lables_from_csv[row[0]]= row[1] #tokenize(row[1])[0] dont need to tokenize
            if len(tokenize(row[3])) == 0:
                count_undone = count_undone + 1 #counts empty descriptipns
                #print(row[0])
            else:
                first_line_true  =  1

    wiki_descrips_dict['schweppes ginger ale'] = 'Ginger ale is a carbonated soft drink flavoured with ginger. It is consumed on its own or used as a mixer, often with spirit-based drinks. There are two main types of ginger ale. The golden style is credited to the Irish doctor, Thomas Joseph Cantrell. The dry style (also called the pale style), a paler drink with a much milder ginger flavour, was created by Canadian John McLaughlin.'
    lables_from_csv['schweppes ginger ale'] = 'mixer'
    wiki_descrips_dict['bulleit® bourbon'] = 'Bulleit Bourbon is a brand of Kentucky straight bourbon whiskey produced at the Kirin Brewing Company Four Roses Distillery in Lawrenceburg, Kentucky, for the Diageo beverage conglomerate. It is characterized by a high rye content for a bourbon and being aged at least six years.'
    lables_from_csv['bulleit® bourbon'] = 'alcohol'
    wiki_descrips_dict['guinness® draught in a can'] = 'Guinness is a dark Irish dry stout that originated in the brewery of Arthur Guinness at St. Jamess Gate, Dublin, Ireland, in 1759. It is one of the most successful beer brands worldwide, brewed in almost 50 countries, and available in over 120. Sales in 2011 amounted to 850 million litres.'
    lables_from_csv['guinness® draught in a can'] = 'alcohol'
    wiki_descrips_dict['green pepper'] = 'The bell pepper is a cultivar group of the species Capsicum annuum. Cultivars of the plant produce fruits in different colours, including red, yellow, orange, green, white, and purple. Bell peppers are sometimes grouped with less pungent pepper varieties as "sweet peppers'
    lables_from_csv['green pepper'] = 'garnish'

    #print(wiki_descrips_dict['ice'])

    #print("this is len of working dict all with descrips", len(wiki_descrips_dict))
    #print(lables_from_csv['blackberry cordial'])
    #print("see if description in here", (wiki_descrips_dict['name'])) it's not
    names = lables_from_csv.keys()

    return wiki_descrips_dict, lables_from_csv, names

def svd_features():
    
    excel_descrip, excel_lab, names = extract_excel_data()
    #make a list of out both the kes (ingred names) and values to pass into future things
    ingred_list = []
    descrip_list = []
    count = 0
    for ingred in excel_descrip:
        ingred_list.append(ingred)
        descrip_list.append(excel_descrip[ingred])

    #print("len ingred list", len(ingred_list))
    #print("len descrip list", len(descrip_list))


    vectorizer = CountVectorizer(tokenizer=tokenize)
    X = vectorizer.fit_transform(descrip_list)

    svd = sklearn.decomposition.TruncatedSVD(n_components=50, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)
    X_reduced = svd.fit_transform(X)  

    #print("new shape", X_reduced.shape)
    #print(svd.singular_values_ )
    #print(X_reduced.shape)


    return X_reduced, excel_lab, names

def do_ml(all_ingredients):

    X_new, X_label_dict, X_names = svd_features()
    X_labels = list(X_label_dict.values())
    X_names_list = list(X_names)

    #shuffle the data
    indices = np.arange(len(X_names_list))
    np.random.shuffle(indices)
    X_shuff = X_new[indices]
    X_labels_shuff = np.asarray(X_labels)[indices]
    X_names_list_shuff = np.asarray(X_names_list)[indices]

    #split into train and test
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_shuff, X_labels_shuff, random_state = 0, shuffle = True)

    svm_model_linear1 = SVC(kernel = 'linear', C = 1).fit(X_train1, y_train1) 
    svm_predictions1 = svm_model_linear1.predict(X_shuff)
    accuracy1 = svm_model_linear1.score(X_shuff, X_labels_shuff)
    #print("svm acc NEW", accuracy1) #better

    knn1 = KNeighborsClassifier(n_neighbors = 20).fit(X_train1, y_train1) #same
    knn_predict = knn1.predict(X_shuff)
    accuracy_knn1 = knn1.score(X_shuff, X_labels_shuff)
    #print("knn acc NEW", accuracy_knn1)

    dtree_model1 = DecisionTreeClassifier(max_depth = 12).fit(X_train1, y_train1) #it maxes accuracy at 5 then plateaus
    dtree_predictions1 = dtree_model1.predict(X_shuff)  #run it using dtree__model.predict
    d_tree_acc = dtree_model1.score(X_shuff, X_labels_shuff)
    #print("dtree acc NEW", d_tree_acc)

    clf1 = GaussianNB()
    naive_bayes1 = clf1.fit(X_train1, y_train1)
    naive_predictions1 = naive_bayes1.predict(X_shuff)
    accuracy_nb1 = naive_bayes1.score(X_shuff, X_labels_shuff)
    #print("nb acc NEW", accuracy_nb1)

    LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train1, y_train1)
    LR_predictions1 = LR.predict(X_shuff)
    accuracy_LR = LR.score(X_shuff, X_labels_shuff)
    #print("LR acc NEW", accuracy_LR)

    neural_network = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1).fit(X_train1, y_train1)
    nn_predict = neural_network.predict(X_shuff)
    accuracy_nn = neural_network.score(X_shuff, X_labels_shuff)
    #print("NN acc NEW", accuracy_nn)

    #ensemble time
    list_classifiers_all = [LR_predictions1, svm_predictions1, nn_predict, dtree_predictions1, knn_predict, naive_predictions1]
    good_classifiers = [LR_predictions1, svm_predictions1, nn_predict, dtree_predictions1] 

    right = 0
    votes = []
    ensamble_predic = {}
    for i in range (0, len(X_names_list)):
        votes.append([LR_predictions1[i], svm_predictions1[i], nn_predict[i], dtree_predictions1[i]])
        majority_vote = stats.mode(votes[i], nan_policy='propagate') #this a runtime warning isn't a problem cause arguments are never NaN
        ingred_at_i = X_names_list_shuff[i]
        ensamble_predic[ingred_at_i] = majority_vote
        if majority_vote[0] == X_labels_shuff[i]:
            right = right + 1
    ensemble_acc = right/len(X_labels_shuff)
    print("ensemble accuracy FINAL", ensemble_acc)


    ml_label_dict = {}
    all_ingreds = copy.deepcopy(all_ingredients)
    for i in range(0, len(all_ingreds)):
        if (all_ingreds[i] not in X_names_list) or all_ingreds[i] == 'smirnoff no. 21® vodka':
            #print("this is in json but not csv (?)", all_ingreds[i]) #these are alcohols that got lost
            ml_label_dict[all_ingreds[i]] = 'alcohol'
        else:
            #if all_ingreds[i] == 'smirnoff no. 21® vodka':
                #ml_label_dict[all_ingreds[i]] = 'alcohol'
            ml_label_dict[all_ingreds[i]] = ensamble_predic[all_ingreds[i]][0][0]

    #print(len(ml_label_dict))

    return ml_label_dict

