from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
from app.irsystem.models.search import *
import os

project_name = "Drink Up!"
net_id = "cjh286, itl5, lm578, sb838, sp772"
cocktail = []

@irsystem.route('/', methods=['GET'])
def search():
	global cocktail
	ingredients = request.args.get('ingredients')
	addToCocktail = request.args.get('add-to-cocktail')
	clearCocktail = request.args.get('clear-cocktail')
	removeFromCocktail = request.args.get('remove-from-cocktail')

	# set up - build all necessary datasets
	drinks_list, all_ingredients_list, recipe_dict, lower_to_upper_i = build_recipe_dict()
	ingredients_dict = build_ingredients_dict(recipe_dict)
	indexTermDict = indexDict(all_ingredients_list)
	co_oc = makeCoOccurrence(recipe_dict, len(all_ingredients_list), indexTermDict[1])

	# user searched ingredients
	if not ingredients:
		rankings = []
		output_message = "Nothing"
	else:
		output_message = "Your Search: " + ingredients
		query = ingredients.split(', ')
		rankings = complementRanking(query, co_oc, indexTermDict[1], indexTermDict[0], lower_to_upper_i)

	# user wants to add item to cocktail
	if addToCocktail:
		addIngredient = getNameFromRanking(addToCocktail)
		if (addIngredient not in cocktail) and (addIngredient != None):
			cocktail.append(addIngredient)

	# user wants to clear the cocktail list
	if clearCocktail:
		cocktail = []

	# user wants to remove an item from the cocktail list
	if removeFromCocktail:
		cocktail.remove(removeFromCocktail)


	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=rankings, cocktail=cocktail)
