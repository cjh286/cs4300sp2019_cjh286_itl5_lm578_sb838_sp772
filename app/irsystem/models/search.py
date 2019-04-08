#this is where you implement your search
import json
from pprint import pprint
import os


def preprocessUK():
    """
    preprocessing function
    """
    script_path = os.path.abspath(__file__) 
    path_list = script_path.split(os.sep)
    script_directory = path_list[0:len(path_list)-4]
    rel_path = 'scraped_data/recipes_uk_output.txt'
    path = "/".join(script_directory) + "/" + rel_path
    
    with open(path) as f:
        data = json.load(f)
    
    data = data['UK_info']

    return data

def randomDrinkNames(data):
    drink_names = []
    for x in range(5):
        drink = data[x]
        drink_names.append(drink['name'])
    
    return drink_names


def drinkNameIndex(data):
    drink_to_index = {}
    index_to_drink = {}
    for d in range(len(data)):
        drink = data[d]
        drink_name = drink['name']
        drink_to_index[drink_name] = d
        index_to_drink[d] = drink_name
    
    return drink_to_index, index_to_drink

def createIngredientsIndex(data):
    for x in data:
        recipe = x['recipe']
        print(x)
        for step in recipe:
            print(step)
        break
    return

if __name__ == "__main__":
    data = preprocessUK()
    # drinkNameIndex = drinkNameIndex(data)
    ingredientsIndex = createIngredientsIndex(data)