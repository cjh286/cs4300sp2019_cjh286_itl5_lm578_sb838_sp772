from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
from app.irsystem.models.search import *
from app.irsystem.models.machine_learning import *
from app.irsystem.models.taste_profiles import *
import os
import copy

# global variables - requires storage
project_name = "Drink Up!"
net_id = "cjh286, itl5, lm578, sb838, sp772"
cocktail = []
rankings = None
output_message = ""
ingredients = None
searchBy = "ingredients"
xable = []

@irsystem.route('/', methods=['GET'])
def search():
	# define global variables
	global cocktail
	global rankings
	global output_message
	global ingredients
	global searchBy
	global xable

	if (request.args.get('ingredients') != None):
		ingredients = request.args.get('ingredients')
	addToCocktail = request.args.get('add-to-cocktail')
	addQueryToCocktail = request.args.get('add-query-to-cocktail')
	clearCocktail = request.args.get('clear-cocktail')
	removeFromCocktail = request.args.get('remove-from-cocktail')
	removeFromQuery = request.args.get('remove-from-query')
	about = request.args.get('about')
	if (request.args.get('search-label') != None):
		searchBy = request.args.get('search-label')
	done = request.args.get('done-cocktail')

	# set up - build all necessary datasets
	drinks_list, all_ingredients_list, recipe_dict, lower_to_upper_i = build_recipe_dict()
	ml_ingred_list = copy.deepcopy(all_ingredients_list)
	ingredients_dict = build_ingredients_dict(recipe_dict)
	indexTermDict = indexDict(all_ingredients_list)
	co_oc = makeCoOccurrence(recipe_dict, len(all_ingredients_list), indexTermDict[1])
	auto_ingredients_list = autoCompleteList(all_ingredients_list)
	labeled_dict = do_ml(ml_ingred_list)
	flavor_dict = create_flavor_dict()

	# user wants to search by
	if not searchBy:
		searchBy = "ingredients"

	# user searched ingredients
	if not ingredients:
		rankings = []
		xable = []
		output_message = ""
	else:
		output_message = "Your Search: " + ingredients
		query = ingredients.split(', ')
		query = queryReformulation(query, all_ingredients_list)
		for q in query:
			if q not in xable:
				xable.append(q)
		initial_rank = complementRanking(xable, co_oc, indexTermDict[1], indexTermDict[0])
		rankings = displayRanking(initial_rank, lower_to_upper_i, labeled_dict, flavor_dict, searchBy)

	# user wants to remove an item from their running query
	if removeFromQuery:
		if (removeFromQuery == 'clear'):
			xable.clear()
		else:
			if xable:
				xable.remove(removeFromQuery)

	# user wants to add the item they queried to the cocktail
	if addQueryToCocktail:
		# TODO: there are some queries not in lower_to_upper_i b/c apostrophes => fix this
		if (addQueryToCocktail in lower_to_upper_i):
			addSearchedIngredient = lower_to_upper_i[addQueryToCocktail]
		else:
			addSearchedIngredient = addQueryToCocktail
		if (addSearchedIngredient not in cocktail) and (addSearchedIngredient != None):
			cocktail.append(addSearchedIngredient)

	# user wants to add item to cocktail
	if addToCocktail:
		# addIngredient = getNameFromRanking(addToCocktail)
		if (addToCocktail not in cocktail) and (addToCocktail != None):
			cocktail.append(addToCocktail)

	# user wants to clear the cocktail list
	if clearCocktail:
		cocktail = []

	# user wants to remove an item from the cocktail list
	if removeFromCocktail:
		cocktail.remove(removeFromCocktail)

	# user is done building recipe
	if done:
		ingred_query = queriesForCocktail(cocktail)
		cocktail_list = makeCocktailRanks(ingred_query, makeJaccard, recipe_dict)
		return render_template('cocktails.html', cocktail=cocktail, cocktail_search = cocktail_list)

	# if xable:
	# 	initial_rank = complementRanking(xable, co_oc, indexTermDict[1], indexTermDict[0])
	# 	rankings = displayRanking(initial_rank, lower_to_upper_i, labeled_dict, flavor_dict, searchBy)

	# takes the user to the about page
	if about:
		return render_template('about.html', name=project_name, netid=net_id)

	# removed searched=ingredients for testing purposes
	return render_template('search.html', name=project_name, netid=net_id, \
		complete_ingredients=json.dumps(auto_ingredients_list), \
			output_message=output_message, data=rankings, cocktail=cocktail, \
				search_by=searchBy, xable=xable)
