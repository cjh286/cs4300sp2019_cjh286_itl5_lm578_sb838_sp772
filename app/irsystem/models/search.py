import json
from pprint import pprint
import os
from collections import defaultdict
import numpy as np
import sys
import copy
import pickle
import operator
import time
# import spacy
# from sklearn.model_selection import train_test_split 
# uncomment line below to test this file only
# from machine_learning import *
# from taste_profiles import *


# nlp = spacy.load('en_core_web_md')


# ======================= Preprocessing Functions ========================
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
    amounts_dict: dict
        Associates recipes with ingredient and amount (list of tuples)
        
    Format
    ======
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
    amounts_dict = {}

    for index in range(num): #this loops through each recipe
        drink_name = data[index].get('name')
        ingredients = data[index].get('ingredients')
        ingredient_list = []
        ingred_amount_list = []
        for i in ingredients:
            ingred = i.get('ingredient')
            amount = i.get('amount')
            lower_to_upper_i[ingred.lower()] = ingred
            ingredient_list.append(ingred.lower())
            if (ingred.lower() not in all_ingredients):
                all_ingredients.append(ingred.lower()) 


            ingred_amount_list.append((ingred.lower(), amount))
        amounts_dict[drink_name.lower()] = ingred_amount_list

        # all_ingredients2 = set(all_ingredients) #this is now the list of ingredients without duplicates
        all_recipes.append(drink_name)
        recipe_dict[drink_name.lower()] = ingredient_list

    return all_recipes, all_ingredients, recipe_dict, lower_to_upper_i, amounts_dict


# set up ingredient list for autocomplete
def autoCompleteList(ingredients_list):
    for x in range(len(ingredients_list)):
        string = ingredients_list[x]
        if ("'" in string):
            ingredients_list[x] = string.replace("'", "`")
    
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


# # ======================= IR System Functions For Ingredient Search ========================
def termDocMatrix(input_dict):
    pass


# create co-occurrence matrix
def makeCoOccurrence(input_dict, n_ingredients, index_dict):
    #print(n_ingredients)
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
    
    return matrix


# TODO: combine complement ranking with display ranking to optimize code
# creates the complement ranking
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
        query_at = query[i].strip()
        query_at_i = query_at.replace("`", "'")
        if (query_at_i in input_term_to_index):
            q_index = input_term_to_index[query_at_i]
            q_column = co_oc_matrix[q_index]
            q__col_normed_list.append(q_column/(co_oc_matrix[q_index][q_index]))
            #all_q_cols[i] = q_column
    for r in range(0,len(q__col_normed_list)):
        q_col_sum = np.add(q_col_sum, q__col_normed_list[r])
    q_col_normed_avg = q_col_sum/len(q__col_normed_list)

    score = sys.maxsize
    numResults = 1
    while (score > 0):
        result = np.argmax(q_col_sum) #gets index
        # if result == q_index: #sets the score of query ingredient with itself to zero
        #     q_col_sum[result] = 0
        score = q_col_sum[result]
        if (score != 0):
            rankeditem = {'item': input_index_to_term[result], 'score': score}
            ranking.append(rankeditem)
            q_col_sum[result] = 0
        numResults += 1
    
    if (len(ranking) == 0):
        return None

    return ranking


# TODO: combine complement ranking with display ranking to optimize code
# intakes a ranking and formats it in a displayable manner
def displayRanking(input_rankings, lower_to_upper, labeled_dict, flavor_dict, search_by):
    print(search_by)
    if (type(input_rankings) != list):
        return "query not found"

    rankings = []
    count = 1
    for x in input_rankings:
        if (x['item'] in labeled_dict):
            label = labeled_dict[x['item']]
        else:
            label = 'n/a'

        # ingred_string = "'{}'".format(x['item'])
        ingred_string = x['item']
        flavor = ''
        if ingred_string not in flavor_dict:
            flavor = 'n/a'
        else:
            for taste_word in flavor_dict[ingred_string]:
                if flavor != '':
                    flavor = flavor + ", " + taste_word
                else:
                    flavor = taste_word

        rankeditem = {'rank': count, 'name': lower_to_upper[x['item']], \
            'score': round(x['score'], 2), 'label': label, 'flavor': flavor}

        if (search_by == "alcohol") or (search_by == "garnish") or (search_by == "mixer"):
            if (search_by == label):
                rankings.append(rankeditem)
                count += 1
        else:
            rankings.append(rankeditem)
            count += 1
        
    if (len(rankings) == 0):
        return "query not found"

    return rankings



# ======================= Cocktail List Display Functions ========================
# TODO: find out if unused and delete if needed
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


# ======================= Query Reformulation ========================
# does wildcard search on queries (so if someone types in vodka, it adds all
# vodkas to the query)
def queryReformulation(og_input_query, input_ingred_list):
    input_query = set(og_input_query)
    new_query = set()
    for q in og_input_query:
        if q in input_ingred_list:
            new_query.add(q)
            input_query.discard(q)
        else:
            # wildcard search - generic to find all brands
            for ingredient in input_ingred_list:
                if q in ingredient:
                    new_query.add(ingredient)
                    input_query.discard(q)
                elif ingredient in q:
                    new_query.add(ingredient)
                    input_query.discard(q)
    
    for q in input_query:
        if q not in new_query:
            new_query.add(q)

    return list(new_query)


# ======================= IR System For Cocktail Search ========================
# formats queries for cocktail search
def queriesForCocktail(input_query):
    new_queries = []

    for q in input_query:
        new_queries.append(q.lower())
    
    return new_queries


# Jaccard used to find similar cocktails
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

    return jacc_dict


# TODO: complete this to show amounts for ingredients
# ranks cocktails based on Jaccard similarity
def makeCocktailRanks(input_query, input_jaccard, input_dict, amounts_dict):
    finalRanks = []
    cocktailJacc = input_jaccard(input_query, input_dict)
    cocktailRanks = sorted(cocktailJacc, key=cocktailJacc.get, reverse=True)

    for x in cocktailRanks:
        ingredients_list = amounts_dict[x]
        score = round(cocktailJacc[x], 2)
        if (score != 0):
            finalRanks.append({'cocktail': x, 'ingredients': ingredients_list, 'score': score})

    return finalRanks


# TODO: complete this
# creates the flavor profile of cocktails
def createCocktailFlavor(input_query, input_flavor_dict):
    # print(input_query)
    cocktail_taste = {}
    for ingred in input_query:
        print(ingred)
        if ingred in input_flavor_dict:
            flavor = input_flavor_dict[ingred]
            print(flavor)
            for x in flavor:
                print(x)
                if x in cocktail_taste:
                    cocktail_taste[x] += 1
                else:
                    cocktail_taste[x] = 1

    ranked = []
    for x in range(len(cocktail_taste)):
        max_flavor = max(cocktail_taste.items(), key=operator.itemgetter(1))[0]
        ranked.append(max_flavor)
        cocktail_taste[max_flavor] = 0

    return ", ".join(ranked)

# testing
def main():
    drinks_list, all_ingredients_list, recipe_dict, lower_to_upper_i, amounts_dict = build_recipe_dict()
    ingredients_dict = build_ingredients_dict(recipe_dict)
    indexTermDict = indexDict(all_ingredients_list)
    co_oc = makeCoOccurrence(recipe_dict, len(all_ingredients_list), indexTermDict[1])
    auto_ingredients_list = autoCompleteList(all_ingredients_list)
    ingred_list_ml = copy.deepcopy(all_ingredients_list)
    labeled_dict = do_ml(ingred_list_ml)
    print(labeled_dict['smirnoff no. 21® vodka'])
    flavor_dict = create_flavor_dict()

    # code for pickling
    with open(r"flavor_dict.pickle", "wb") as output_file:
        pickle.dump(flavor_dict, output_file)

    # test queries
    query = ['mincemeat', 'cranberry juice', 'orange juice']
    search_by = 'ingredients'
    query = queryReformulation(query, all_ingredients_list)
    rankings = complementRanking(query, co_oc, indexTermDict[1], indexTermDict[0])[:10]
    # ranked = displayRanking(rankings, lower_to_upper_i, labeled_dict, flavor_dict, search_by)
    
    query1 = ['baileys® original irish cream liqueur', 'caster sugar', 'dark chocolate', \
        'cocoa powder', 'sweet oat biscuits']
    cocktail_ranks = makeCocktailRanks(query, makeJaccard, recipe_dict, amounts_dict)
    # print(cocktail_ranks)

    #print(createCocktailFlavor(query1, flavor_dict))
   
if __name__ == "__main__":
    main()
