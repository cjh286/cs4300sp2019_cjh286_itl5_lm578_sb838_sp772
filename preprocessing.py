import json
from collections import defaultdict

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

    
def build_recipe_dict(data, num):
    """ Create dictionary from recipe data of recipe names and 
        corresponding list of ingredients
        Create lists of all recipes and all ingredients.
        
        NOTE: should we cast all information to lowercase?
    
    Arguments
    =========
    
    data: list of dicts
        Each entry (dictionary) contains the same set of fields
    num: int
        The number of recipes contained within the data
        
    Returns
    =======
    
    all_recipes: list
        List of all drink names in the data
    all_ingredients: set
        Set of all unique ingredients in the data
    recipe_dict: dict
        Each entry to the dictionary consists of a key (recipe name)
        and a list of ingredients as the corresponding value
        
    Format
    ======
    
        { 
         recipe name: [ingredient1, ingredient2, ...],
         recipe name: [ingredient1, ingredient2, ...],
         ...
       }

    """
    
    recipe_dict = defaultdict(list)
    all_recipes = []
    all_ingredients = []
    
    for index in range(num):
        drink_name = data[index].get('name')
        ingredients = data[index].get('ingredients')
        ingredient_list = [i.get('ingredient') for i in ingredients] 

        all_recipes.append(drink_name)
        all_ingredients.extend(ingredient_list)
        recipe_dict[drink_name] = ingredient_list
    
    return all_recipes, set(all_ingredients), recipe_dict


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


def main():
    with open("scraped_data/uk_output.json") as f:
        data = json.loads(f.readlines()[0])
    num_recipes = len(data)
    print(type(data))
    
    ### print some information about the json file ###
    setup(data, num_recipes)
    
    ### collect lists of all recipes and ingredients ###
    ### create dictionary containing recipe names and list of ingredients ###
    drinks_list, ingredients_list, recipe_dict = build_recipe_dict(data, num_recipes)
    
    ### build dictionary of ingredients to recipes ###
    ingredients_dict = build_ingredients_dict(recipe_dict)
    print(ingredients_dict["Lemon Juice"])
    
if __name__ == "__main__":
    main()