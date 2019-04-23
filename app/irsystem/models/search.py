import json
from pprint import pprint
import os
from collections import defaultdict
import numpy as np
import sys
import copy
from sklearn.model_selection import train_test_split 
# from machine_learning import *
# import spacy

# nlp = spacy.load('en_core_web_md')


# preprocessing functions
def setup(data, num):
    """ Print some basic information about the json file being parsed

    Arguments
    =========

    data: list of dicts
        Each entry (dictionary) contains the same set of fields
    num: int
        The number of recipes contained within the data

    """   
    
    print("Loaded {} recipes".format(num))
    print("Each recipe is a dictionary with the following keys...")
    print(data[0].keys())
    print("--------------------------")

    
def build_recipe_dict():
    """ Create dictionary from recipe data of recipe names and 
        corresponding list of ingredients
        Create lists of all recipes and all ingredients.
        
        NOTE: should we cast all information to lowercase? -- 
              dictionaries go to lowercase, recipe and ingredient list
              are still uppercase

    Returns
    =======
    
    all_recipes: list
        List of all drink names in the data
    all_ingredients: set
        Set of all unique ingredients in the data
    recipe_dict: dict
        Each entry to the dictionary consists of a key (recipe name)
        and a list of ingredients as the corresponding value
    lower_to_upper_i: dict
        Associates lower and uppercase of each ingredient
        
    Format
    ======
    
        { 
         recipe name: [ingredient1, ingredient2, ...],
         recipe name: [ingredient1, ingredient2, ...],
         ...
       }

    """
    script_path = os.path.abspath(__file__) 
    path_list = script_path.split(os.sep)
    script_directory = path_list[0:len(path_list)-4]
    rel_path = 'scraped_data/uk_output.json'
    path = "/".join(script_directory) + "/" + rel_path
    
    with open(path) as f:
        data = json.loads(f.readlines()[0])
    num = len(data)

    recipe_dict = defaultdict(list)
    all_recipes = []
    all_ingredients = []
    lower_to_upper_i = {}
    
    for index in range(num): #this loops through each recipe
        drink_name = data[index].get('name')
        ingredients = data[index].get('ingredients')
        ingredient_list = []
        for i in ingredients:
            ingred = i.get('ingredient')
            lower_to_upper_i[ingred.lower()] = ingred
            ingredient_list.append(ingred.lower())
            all_ingredients.append(ingred.lower()) #this is the list of ingredients with duplicates
        
        all_ingredients2 = set(all_ingredients) #this is now the list of ingredients without duplicates
        all_recipes.append(drink_name)
        recipe_dict[drink_name.lower()] = ingredient_list
    
    # print(len(all_ingredients))
    # print(len(all_ingredients2))

    return all_recipes, list(all_ingredients2), recipe_dict, lower_to_upper_i 


def autoCompleteList(ingredients_list):
    for x in range(len(ingredients_list)):
        string = ingredients_list[x]
        ingredients_list[x] = string.replace("'", "&#39;")
    
    return (ingredients_list)


def build_ingredients_dict(input_dict):
    """ build term-frequency of ingredients to recipes
    
    Arguments
    =========
    
    input_dict: dict
        The recipe dictionary of recipes to ingredients
    
    Returns
    =======
    
    ingredient_freq: dict
        Each entry has an ingredient key and a corresponding list of recipes
        that the ingredient appears in

    """
    
    output = defaultdict(list)
    
    for recipe in input_dict.keys():
        i_list = input_dict.get(recipe)
        for i in i_list:
            output[i].append(recipe)
            
    return output


def indexDict(input_list):
    """
    Creates dictionaries that map each ingredient to an index
    Inputs:
        input_list: list of ingredients
    Outputs:
        indexToTerm: {index: ingredient}
        termToIndex: {term: index}
    """
    input_list = list(input_list)
    indexToTerm = {}
    termToIndex = {}
    
    for x in range(len(input_list)):
        indexToTerm[x] = input_list[x]
        termToIndex[input_list[x]] = x
    
    return indexToTerm, termToIndex


# search functions
def termDocMatrix(input_dict):
    pass


def makeCoOccurrence(input_dict, n_ingredients, index_dict):
    """
    Inputs: 
        input_dict: recipe_dict (recipes: list of ingredients)
        n_ingredients: total number of ingredients
        index_dict: term to index dictionary
    Outputs:
        n x n matrix: ingredient by ingredient matrix
        matrix[i][j] is number of co-occurrences of of ingredient i and ingredient j
    
    A co-occurrence is when two ingredients appear in the same recipe.
    """
    matrix = np.zeros((n_ingredients, n_ingredients))
    # tupleDict was for list of recipes of each co-occurrence, not using it at the moment
    # tupleDict = {} 
    
    for x in input_dict:
        for i1 in range(0, len(input_dict[x])):
            for i2 in range(i1, len(input_dict[x])):
                ingredient1 = input_dict[x][i1]
                ingredient2 = input_dict[x][i2]
                index1 = index_dict[ingredient1]
                index2 = index_dict[ingredient2]
                if index1 == index2:
                    matrix[index1][index2] += 1
                elif (index1 != index2):
                    matrix[index1][index2] += 1
                    matrix[index2][index1] += 1
                    # if ((ingredient1, ingredient2) not in tupleDict):
                    #     tupleDict[(ingredient1, ingredient2)] = set([x])
                    #     tupleDict[(ingredient2, ingredient1)] = set([x])
                    # else:
                    #     tupleDict[(ingredient1, ingredient2)].add(x)
                    #     tupleDict[(ingredient2, ingredient1)].add(x)
    
    return matrix


def complementRanking(query, co_oc, input_term_to_index, input_index_to_term):
    """
    Create ranking of complements based on query
    Inputs:
        query: list of string (that user searches)
        co_oc_matrix: co-occurrence matrix created in earlier function
        input_term_to_index: term to index matrix
        input_index_to_term: index to term matrix
        lower_to_upper: dictionary mapping lower case ingredient strings to upper case (original format)
    Outputs:
        ranking: list of strings, formatted as ranked number. ingredient (score: score number)
    """

    co_oc_matrix = copy.deepcopy(co_oc)
    ranking = []

    q_col_sum = np.zeros(len(input_term_to_index))
    q__col_normed_list = list()
    q_col_averaged = np.zeros(len(input_term_to_index))
    for i in range (len(query)):
        query_at_i = query[i].strip()
        if (query_at_i in input_term_to_index):
            q_index = input_term_to_index[query_at_i]
            q_column = co_oc_matrix[q_index]
            q__col_normed_list.append(q_column/(co_oc_matrix[q_index][q_index]))
            #all_q_cols[i] = q_column
    for r in range(0,len(q__col_normed_list)):
        q_col_sum = np.add(q_col_sum, q__col_normed_list[i])
    q_col_normed_avg = q_col_sum/len(q__col_normed_list)

    score = sys.maxsize
    numResults = 1
    while (score > 0):
        result = np.argmax(q_col_sum) #gets index
        score = q_col_sum[result]
        if (score != 0):
            rankeditem = {'rank': numResults, 'item': input_index_to_term[result], 'score': score}
            ranking.append(rankeditem)
            q_col_sum[result] = 0
        numResults += 1
  
    if (len(ranking) == 0):
        return "query not found"

    return ranking


def displayRanking(input_rankings, lower_to_upper, labeled_dict):
    rankings = []

    for x in input_rankings:
        print(x)
        if (x['item'] in labeled_dict):
            label = labeled_dict[x['item']]
        else:
            label = 'n/a'
        rankeditem = {'rank': x['rank'], 'name': lower_to_upper[x['item']], 'score': round(x['score'], 2), 'label': label}
        rankings.append(rankeditem)

    return rankings



# making cocktail list functions
def getNameFromRanking(rankedInput):
    """
    Gets just the name of the ingredient from the ranking outputs
    Inputs:
        rankedInput: one item from the rankings list
    Outputs:
        name: ingredient name from input
    """
    periodIndex = rankedInput.find('.')
    paranIndex = rankedInput.find('(')
    if (periodIndex != -1) and (paranIndex != -1):
        name = rankedInput[periodIndex+2:paranIndex-1]
    else: 
        name = None
    return name


# misc functions
def queryReformulation(input_query, input_ingred_list):
    new_query = []
    for q in input_query:
        if q in input_ingred_list:
            new_query.append(q)
        else:
            # wildcard search - generic to find all brands
            for ingredient in input_ingred_list:
                pass
            

    return new_query

def makeJaccard(input_query, input_dict):
    #######
    query = list()
    query.extend(input_query)
    query_set = set(query)
    jacc_dict=dict()
    ingreds_common_dict = dict() #the ingredients in common between query and recipe

    for drink in input_dict:
        ingreds_set = set(input_dict[drink])

        intersect = set()
        union = set()
        intersect = ingreds_set.intersection(query_set)
        union = ingreds_set.union(query_set)
        ingreds_common_dict[drink] = intersect
        len_in = len(intersect)
        len_un = len(union)

        if len_un > 0:
            jacc_dict[drink] = len_in/len_un
        else:
            jacc_dict[drink] = 0

    
    list_sort = (sorted(jacc_dict, key=jacc_dict.get, reverse=True)[:10])

    #print first 10
    
    for i in range(0,10):
        recipe_name = list_sort[i]
        #THIS PRINTS TOP TEN CORRECTLY IF UNCOMMENTED
        #print(i+1 , recipe_name, "\t\tJaccard score:",round(jacc_dict[recipe_name],2), "\tIngredients in common:", ingreds_common_dict[recipe_name])
    #print("HERE",list_sort)
    return list_sort

def make_features(ingred_list):

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
            print(x, "an unlabled ingredient was found") 
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



# testing
def main():
    ### collect lists of all recipes and ingredients ###
    ### create dictionary containing recipe names and list of ingredients and lowercase-uppercase associations dictionary ###
    drinks_list, all_ingredients_list, recipe_dict, lower_to_upper_i = build_recipe_dict()
    # print("len drinks list:", len(drinks_list), "len all ingredients list:", len(all_ingredients_list))

    ### build dictionary of ingredients to recipes ###
    ingredients_dict = build_ingredients_dict(recipe_dict)

    # build dictionaries for indexes to terms
    indexTermDict = indexDict(all_ingredients_list)
    
    ### build co-occurrence matrix ###
    co_oc = makeCoOccurrence(recipe_dict, len(all_ingredients_list), indexTermDict[1])

    #creates a labeled dictionary for alcohol/mixer/garnish where keys are ingredient names, values are labels
    labeled_dict = do_ml(all_ingredients_list)


    # test queries
    query = ['orange juice']
    query2 = ['cranberry juice']
    query3 = ['cranberry juice', 'orange juice']
    rankings1 = complementRanking(query, co_oc, indexTermDict[1], indexTermDict[0])
    rankings2 = complementRanking(query2, co_oc, indexTermDict[1], indexTermDict[0])
    rankings3 = complementRanking(query3, co_oc, indexTermDict[1], indexTermDict[0])
    # print(rankings1[:10])
    # print("")
    # print(rankings2[:10])
    # print("")
    # print(rankings3[:10])

    # print(all_ingredients_list)
    # print(rankings1)

    # display = displayRanking(rankings1, lower_to_upper_i, labeled_dict)
    # print(display)

if __name__ == "__main__":
    main()