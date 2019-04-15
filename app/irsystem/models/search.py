#this is where you implement your search
import json
from pprint import pprint
import os
from collections import defaultdict
import numpy as np
import sys

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

    return all_recipes, all_ingredients2, recipe_dict, lower_to_upper_i 

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
            
#    print(output["Lemon Juice"])
    return output


def termDocMatrix(input_dict):
    pass


def makeCoOccurrence(input_dict, n_ingredients, index_dict):
    #n x n matrix -> ingredient by ingredient matrix
    matrix = np.zeros((n_ingredients, n_ingredients))
    tupleDict = {}
    
    for x in input_dict:
        #x is each recipe
        # print(x)
        # print(input_dict[x])
        for i1 in range(0, len(input_dict[x])):
            for i2 in range(i1, len(input_dict[x])):
                ingredient1 = input_dict[x][i1]
                ingredient2 = input_dict[x][i2]
                index1 = index_dict[ingredient1]
                index2 = index_dict[ingredient2]
                if (index1 != index2):
                    matrix[index1][index2] += 1
                    matrix[index2][index1] += 1
                    if ((ingredient1, ingredient2) not in tupleDict):
                        tupleDict[(ingredient1, ingredient2)] = set([x])
                        tupleDict[(ingredient2, ingredient1)] = set([x])
                    else:
                        tupleDict[(ingredient1, ingredient2)].add(x)
                        tupleDict[(ingredient2, ingredient1)].add(x)
    
    return matrix

def indexDict(input_list):
    input_list = list(input_list)
    indexToTerm = {}
    termToIndex = {}
    
    for x in range(len(input_list)):
        indexToTerm[x] = input_list[x]
        termToIndex[input_list[x]] = x
    
    return indexToTerm, termToIndex


def complementRanking(query, co_oc_matrix, input_term_to_index, input_index_to_term):
    ranking = []
    if (len(query) == 1):
        if (query[0] in input_term_to_index):
            q_index = input_term_to_index[query[0]]
            q_column = co_oc_matrix[q_index]

            score = sys.maxsize
            numResults = 1
            while (score > 0):
                result = np.argmax(q_column)
                score = q_column[result]
                if (score != 0):
                    rankeditem = str(numResults) + ". " + input_index_to_term[result] + " (score: " + str(score) + ")"
                    ranking.append(rankeditem)
                    q_column[result] = 0
                numResults += 1
        else:
            ranking.append("query not found")
    else:
        pass
    
    return ranking


def getNameFromRanking(rankedInput):
    periodIndex = rankedInput.find('.')
    paranIndex = rankedInput.find('(')
    if (periodIndex != -1) and (paranIndex != -1):
        name = rankedInput[periodIndex+2:paranIndex-1]
    else: 
        name = None
    return name


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

def main():
    # script_path = os.path.abspath(__file__) 
    # path_list = script_path.split(os.sep)
    # script_directory = path_list[0:len(path_list)-4]
    # rel_path = 'scraped_data/uk_output.json'
    # path = "/".join(script_directory) + "/" + rel_path
    # with open(path) as f:
    #     data = json.loads(f.readlines()[0])
    # num_recipes = len(data)
    # print(type(data))
    
    # ### print some information about the json file ###
    # setup(data, num_recipes)

    ### collect lists of all recipes and ingredients ###
    ### create dictionary containing recipe names and list of ingredients and lowercase-uppercase associations dictionary ###
    drinks_list, all_ingredients_list, recipe_dict, lower_to_upper_i = build_recipe_dict()
    print("len drinks list:", len(drinks_list), "len all ingredients list:", len(all_ingredients_list))

    ### build dictionary of ingredients to recipes ###
    ingredients_dict = build_ingredients_dict(recipe_dict)

    
    indexTermDict = indexDict(all_ingredients_list)
    
    ### build co-occurrence matrix ###
    co_oc = makeCoOccurrence(recipe_dict, len(all_ingredients_list), indexTermDict[1])

    query = ['orange juice']
    rankings = complementRanking(query, co_oc, indexTermDict[1], indexTermDict[0])
    print(getNameFromRanking("fjeiow"))

# for testing only
if __name__ == "__main__":
    main()