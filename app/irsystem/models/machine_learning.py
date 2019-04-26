#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
import numpy as np
import copy
from sklearn.model_selection import train_test_split 
# import spacy

# nlp = spacy.load('en_core_web_md')
# # tokens = nlp(u'dog cat banana afskfsd')
# # for token in tokens:
# #     print(token.text, token.has_vector, token.vector_norm, token.is_oov)

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

    from sklearn.svm import SVC 
    svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
    svm_predictions = svm_model_linear.predict(X_test)
    accuracy = svm_model_linear.score(X_test, y_test)
    print("svm acc", accuracy)

    from sklearn.neighbors import KNeighborsClassifier 
    knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train) 
  
    # accuracy on X_test 
    accuracy_knn = knn.score(X_test, y_test) 
    print("knn acc", accuracy_knn)

    #make the dtree model
    from sklearn.tree import DecisionTreeClassifier 
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
    from sklearn.naive_bayes import GaussianNB
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
