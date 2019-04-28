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
            else:
                first_line_true  =  1

    print("this is len of working dict all with descrips", len(wiki_descrips_dict))
    print("this many ingreds in csv without any description:", count_undone)
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


def make_features(input_ingred_list):

    ingred_list = copy.deepcopy(input_ingred_list)

    num_features = 21
    feature_np = np.zeros((len(ingred_list),num_features))

    for i in range(0,len(ingred_list)):
        feat_num = 0
        if len(ingred_list[i])>= 4:
            feature_np[i][0] = 1
        feat_num = 1
        if "vo" in ingred_list[i]:
            feature_np[i][1] = 1
        feat_num = 2
        if "lemon" in ingred_list[i]:
            feature_np[i][2] = 1
        feat_num = 3
        if "garnish" in ingred_list[i]:
            feature_np[i][3] = 1
        feat_num = 4
        if "gin " in ingred_list[i]:
            feature_np[i][4] = 1
        feat_num = 5
        if "sy" in ingred_list[i]:
            feature_np[i][5] = 1
        feat_num = 6
        if "liq" in ingred_list[i]:
            feature_np[i][6] = 1
        feat_num = 7
        if "blen" in ingred_list[i]:
            feature_np[i][7] = 1
        feat_num = 8
        if "®" in ingred_list[i]:
            feature_np[i][8] = 1
        feat_num = 9
        if "cho" in ingred_list[i]:
            feature_np[i][9] = 1
        feat_num = 10
        if "bit" in ingred_list[i]:
            feature_np[i][10] = 1
        feat_num = 11
        if "gro" in ingred_list[i]:
            feature_np[i][11] = 1
        feat_num = 12
        if "wat" in ingred_list[i]:
            feature_np[i][12] = 1
        feat_num = 13
        if "mint" in ingred_list[i]:
            feature_np[i][13] = 1
        feat_num = 14
        if "ju" in ingred_list[i]:
            feature_np[i][14] = 1
        feat_num = 15
        if "z" in ingred_list[i]:
            feature_np[i][15] = 1
        feat_num = 16
        if "slic" in ingred_list[i]:
            feature_np[i][16] = 1
        feat_num = 17
        if "rum" in ingred_list[i]:
            feature_np[i][17] = 1
        feat_num = 18
        if "whis" in ingred_list[i]:
            feature_np[i][18] = 1
        feat_num = 19
        if "fru" in ingred_list[i]:
            feature_np[i][19] = 1
        feat_num = 20
        if "flo" in ingred_list[i]:
            feature_np[i][20] = 1
        feat_num = 21
        # token = nlp(u'ingred_list[i]')[0]
        # if token.has_vector:
        #     feature_np[i][21] = ingred_list.vector_norm

    return feature_np

def do_ml(all_ingredients):

    X_new, X_label_dict, X_names = svd_features()
    X_labels = list(X_label_dict.values())
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_new, X_labels, random_state = 0, shuffle = True)
    #how to match x name with its column later to get the index since it's shuffled????

    
    svm_model_linear1 = SVC(kernel = 'linear', C = 1).fit(X_train1, y_train1) 
    svm_predictions1 = svm_model_linear1.predict(X_test1)
    accuracy1 = svm_model_linear1.score(X_test1, y_test1)
    #print("svm acc NEW", accuracy1) #better

    knn1 = KNeighborsClassifier(n_neighbors = 20).fit(X_train1, y_train1) #same
    knn_predict = knn1.predict(X_test1)
    accuracy_knn1 = knn1.score(X_test1, y_test1) 
    #print("knn acc NEW", accuracy_knn1)

    dtree_model1 = DecisionTreeClassifier(max_depth = 12).fit(X_train1, y_train1) #it maxes accuracy at 5 then plateaus
    dtree_predictions1 = dtree_model1.predict(X_test1)  #run it using dtree__model.predict
    d_tree_acc = dtree_model1.score(X_test1, y_test1)
    #print("dtree acc NEW", d_tree_acc)

    clf1 = GaussianNB()
    naive_bayes1 = clf1.fit(X_train1, y_train1)
    naive_predictions1 = naive_bayes1.predict(X_test1)
    accuracy_nb1 = naive_bayes1.score(X_test1, y_test1)
    #print("nb acc NEW", accuracy_nb1)

    LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train1, y_train1)
    LR_predictions1 = LR.predict(X_test1)
    accuracy_LR = LR.score(X_test1, y_test1)
    #print("LR acc NEW", accuracy_LR)

    neural_network = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1).fit(X_train1, y_train1)
    nn_predict = neural_network.predict(X_test1)
    accuracy_nn = neural_network.score(X_test1, y_test1)
    #print("NN acc NEW", accuracy_nn)

    #ensamble time
    list_classifiers_all = [LR_predictions1, svm_predictions1, nn_predict, dtree_predictions1, knn_predict, naive_predictions1]
    good_classifiers = [LR_predictions1, svm_predictions1, nn_predict, dtree_predictions1] 

    right = 0
    votes = []
    for i in range (0, len(X_test1)):
        votes.append([LR_predictions1[i], svm_predictions1[i], nn_predict[i], dtree_predictions1[i]])
        majority_vote = stats.mode(votes[i]) #why is this a runtime warning
        #ensamble_label_dict[ingredNNNNNNN????????] = majority_vote
        if majority_vote[0] == y_test1[i]:
            right = right + 1
    ensamble_acc = right/len(X_test1)
    print("ensamble accuracy FINAL", ensamble_acc)


    all_list1 = all_ingredients
    all_list2 = copy.deepcopy(all_ingredients)


    ingred_np_train = np.array(['orchid','cinnamon powder','lemon zest','peach bitters','pimm’s® strawberry with a hint of mint',
        'angostura aromatic bitters','pimento dram','sweet vermouth','original bitters','tonic water','baileys® coffee irish cream liqueur',
        'raspberry','almond syrup','blackberry cordial','hot water','mascarpone','nutmeg','aromatic bitters','ginger','crème de framboise',
        'smirnoff® vanilla flavoured vodka','ice','ground cinnamon','orange bitters','angostura bitters','agave nectar','orange',
        'dark chocolate','demerara sugar syrup','gordon’s london dry gin','oreo cookies','diet tonic water','rosemary','sprig of dill',
        'apple','lime','elderflower syrup','olive brine','strawberry','elderflower','milk','olive','pickle vinegar','passion fruit juice',
        'mincemeat','egg yolk','chocolate popping candy','boiling water','grated nutmeg','calvados','candy cane',
        'mint to garnish','lime twist','cracked black pepper','ruby port','streaky bacon','crushed biscuit','dry vermouth','cola',
        'asperol','mango pureé','elderflower cordial','ginger beer','tanqueray® london dry gin',
        'johnnie walker® platinum blended scotch whisky','lavender','smoked salt','smirnoff® espresso flavoured vodka','coffee bean',
        'j&b rare® blended scotch whisky','hundreds-and-thousands','cranberry jelly','ginger juice','watermelon juice','cucumber water',
        'captain morgan® original spiced rum','tomato juice','raspberry liqueur','beets and veg juice','soda','apple cider','sour syrup',
        'pedro ximenez -sherry-','sparkling water','red grape','guinness® draught in a can','lemon to garnish','schweppes tonic water',
        'fresh chopped strawberries','pear oak bitters','cardamom','bitters','pink lemonade','ginger bitters','caramelised onion chutney',
        'egg','lemon twist and mint sprig to garnish','orange peel','chocolate bitters','grape soda','dark chocolate at least 75% cocoa',
        'chocolate syrup','pomegranate seeds','sparkling apple soda','guava juice','baileys® pumpkin spice','apple juice','rose syrup',
        'lemon sherbet','raspberries','brussel sprouts','lime juice','baileys® original irish cream liqueur','talisker® storm malt whisky',
        'lemon wheel','red pepper','apricot brandy','split vanilla pod','fresh ginger','iced tea','ketel one citroen® flavored vodka','grape',
        'rose petal','carbonated water','wasabi','coriander seed syrup','apricot nectar','cherry','coconut water','basil leaves',
        'gummy worms sweets','smirnoff® blueberry flavoured vodka','orange slice','white chocolate liqueur','orange extract','celery salt',
        'hot milk','marigold flower','honey syrup','sage leaves','dark chocolate ganache','sweet violets','baileys original (0.8 units)',
        'banana','raw apple cider vinegar','distilled vinegar','campari','cherry syrup','lemon wedge'])

    #print("here",ingred_np_train[49], ingred_np_train[60])

    #labels for training set
    labels_list_train = np.array(['garnish','garnish','garnish','alcohol','alcohol','alcohol','alcohol','alcohol','alcohol','mixer','alcohol',
        'garnish','mixer','alcohol','mixer','mixer','garnish','alcohol','garnish','alcohol','alcohol','garnish','garnish','alcohol','alcohol',
        'mixer','garnish','garnish','mixer','alcohol','garnish','mixer','garnish','garnish','garnish','garnish','mixer','mixer','garnish','garnish',
        'mixer','garnish','mixer','mixer','garnish','mixer','garnish','mixer','garnish','alcohol','garnish','garnish','garnish','garnish',
        'alcohol','garnish','garnish','alcohol','mixer','alcohol','mixer','alcohol','alcohol','alcohol','alcohol','garnish','garnish',
        'alcohol','garnish','alcohol','garnish','mixer','mixer','mixer','mixer','alcohol','mixer','alcohol','mixer','mixer','mixer','mixer','alcohol',
        'mixer','garnish','alcohol','garnish','mixer','garnish','alcohol','garnish','alcohol','mixer','alcohol','mixer','mixer','garnish','garnish',
        'alcohol','mixer','garnish','garnish','garnish','mixer','mixer','alcohol','mixer','mixer','mixer','garnish','garnish','mixer','garnish',
        'garnish','garnish','garnish','alcohol','garnish','garnish','mixer','acohol','garnish','garnish','mixer','mixer','mixer','mixer','garnish',
        'mixer','garnish','garnish','alcohol','garnish','alcohol','mixer','garnish','mixer','garnish','mixer','garnish','garnish','garnish','alcohol',
        'garnish','mixer','mixer','alcohol','mixer','garnish'])


    #make features for the pre-labled ingredients
    list_training_ingreds = ingred_np_train.tolist()
    features_for_train = make_features(list_training_ingreds)
    
    #split the labeled data into train and test
    X_train, X_test, y_train, y_test = train_test_split(features_for_train, labels_list_train, random_state = 0) 

    
    svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
    svm_predictions = svm_model_linear.predict(X_test)
    accuracy = svm_model_linear.score(X_test, y_test)
    print("svm acc", accuracy)

    
    knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train) 
    accuracy_knn = knn.score(X_test, y_test) 
    print("knn acc", accuracy_knn)

    #make the dtree model
    dtree_model = DecisionTreeClassifier(max_depth = 5).fit(X_train, y_train) #it maxes accuracy at 5 then plateaus
    dtree_predictions = dtree_model.predict(X_test)  #run it using dtree__model.predict

    #get the accuracy of the model
    right = 0
    for j in range(0,len(y_test)):
        if y_test[j]==dtree_predictions[j]:
            right = right +1
        #else:
            #print("got wrong:", ingred_np_train[j])
    print("dtree acc", right/len(y_test))

    #naive bayes
    
    clf = GaussianNB()
    naive_bayes = clf.fit(X_train, y_train)
    naive_predictions = naive_bayes.predict(X_test)
    accuracy_nb = naive_bayes.score(X_test, y_test)
    print("nb acc", accuracy_nb)

    #now that the model has been trained, run it on the remaining 339 ingredients to label them:
    for x in list_training_ingreds:
        if x in all_list1:
            all_list1.remove(x) #this removes from the list of all ingredients, the ones that were trained on
        else:
            pass
            # print(x, "an unlabled ingredient was found") 
            #this is to make sure all the names match so if a trained name is not found in all ingredients it will print the name
    untrained_ingreds = all_list1
    untrained_ingreds_np = np.asarray(all_list1)
    #print("len untrained", len(untrained_ingreds)) #check that this is 339

    features_untrained = make_features(untrained_ingreds)
    dtree_predictions2 = dtree_model.predict(features_untrained)
    #dtree_predictions2 are the machine learning-labeled ingredients

    #return a dictionary
    label_dic = dict()
    for i in range(0,len(all_list2)):
        if all_list2[i] in ingred_np_train:
            #print(all_list2[i])
            idx = np.argwhere(ingred_np_train == all_list2[i])
            label_dic[all_list2[i]] = labels_list_train[idx[0][0]]
        elif all_list2[i] in untrained_ingreds:
            idx = np.argwhere(untrained_ingreds_np == all_list2[i])
            label_dic[all_list2[i]] = dtree_predictions2[idx[0][0]]
        else:
            print(all_ingredients[i], "has no label, something has gone wrong")


    return label_dic

# labeled_dict = do_ml(all_ingredients_list)
