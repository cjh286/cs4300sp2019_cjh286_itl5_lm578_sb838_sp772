from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
from app.irsystem.models.search import *
from app.irsystem.models.machine_learning import *
from app.irsystem.models.taste_profiles import *
import os
import copy
import pickle
# import time

# global variables - requires storage
project_name = "Drink Up!"
net_id = "cjh286, itl5, lm578, sb838, sp772"
search_by_options = ['alcohol', 'mixer', 'garnish']

@irsystem.route('/', methods=['GET'])
def search():
	# start_time = time.time()
	global search_by_options
	if ('cocktail' not in session):
		session['cocktail'] = []
	if ('output_message' not in session):
		session['output_message'] = ""
	if ('searchBy' not in session):
		session['searchBy'] = 'ingredients'
	if ('xable' not in session):
		session['xable'] = []
	if ('rankings' not in session):
		session['rankings'] = []


	ingredients = request.args.get('ingredients')
	# print(ingredients == "")
	# print(ingredients)
	addToCocktail = request.args.get('add-to-cocktail')
	addQueryToCocktail = request.args.get('add-query-to-cocktail')
	clearCocktail = request.args.get('clear-cocktail')
	removeFromCocktail = request.args.get('remove-from-cocktail')
	removeFromQuery = request.args.get('remove-from-query')
	about = request.args.get('about')
	if (request.args.get('search-label') in search_by_options):
		session['searchBy'] = request.args.get('search-label')
	else:
		session['searchBy'] = 'ingredients'
	done = request.args.get('done-cocktail')

	# drinks_list, all_ingredients_list, recipe_dict, lower_to_upper_i = build_recipe_dict()
	# ingredients_dict = build_ingredients_dict(recipe_dict)
	# indexTermDict = indexDict(all_ingredients_list)
	# co_oc = makeCoOccurrence(recipe_dict, len(all_ingredients_list), indexTermDict[1])
	# auto_ingredients_list = autoCompleteList(all_ingredients_list)
	# ingred_list_ml = copy.deepcopy(all_ingredients_list)
	# labeled_dict = do_ml(ingred_list_ml)
	# flavor_dict = create_flavor_dict()
	
	# set up dictionaries
	with open(r'pickle/drinks_list.pickle', "rb") as input_file:
			drinks_list = pickle.load(input_file)
	with open(r'pickle/all_ingredients_list.pickle', "rb") as input_file:
			all_ingredients_list = pickle.load(input_file)
	with open(r'pickle/recipe_dict.pickle', "rb") as input_file:
			recipe_dict = pickle.load(input_file)
	with open(r'pickle/lower_to_upper_i.pickle', "rb") as input_file:
		lower_to_upper_i = pickle.load(input_file)
	with open(r'pickle/ingredients_dict.pickle', "rb") as input_file:
		ingredients_dict = pickle.load(input_file)
	with open(r'pickle/indexTermDict.pickle', "rb") as input_file:
		indexTermDict = pickle.load(input_file)
	with open(r'pickle/co_oc.pickle', "rb") as input_file:
		co_oc = pickle.load(input_file)
	with open(r'pickle/auto_ingredients_list.pickle', "rb") as input_file:
		auto_ingredients_list = pickle.load(input_file)
	with open(r'pickle/labeled_dict.pickle', "rb") as input_file:
		labeled_dict = pickle.load(input_file)
	with open(r'pickle/flavor_dict.pickle', "rb") as input_file:
		flavor_dict = pickle.load(input_file)

	# user searched ingredients
	if (ingredients == ""):
		session['rankings'] = []
		session['xable'] = []
		session['output_message'] = ""
		session.modified = True
		# session['ingredients'] = None
	else:
		if (ingredients != None):
			rankings = session.pop('rankings', [])
			output_message = "Your Search:"
			query = ingredients.split(', ')
			query = queryReformulation(query, all_ingredients_list)
			xable = session.pop('xable', [])
			for q in query:
				if q not in xable:
					xable.append(q)
			initial_rank = complementRanking(xable, co_oc, indexTermDict[1], indexTermDict[0])
			rankings = displayRanking(initial_rank, lower_to_upper_i, labeled_dict, flavor_dict, session['searchBy'])[:50]
			
			session['rankings'] = rankings
			session['xable'] = xable


	# user wants to remove an item from their running query
	if removeFromQuery:
		xable = session.pop('xable', [])
		if (removeFromQuery in xable):
			xable.remove(removeFromQuery)
			if (len(xable) == 0):
				session['rankings'] = []
			else:
				initial_rank = complementRanking(xable, co_oc, indexTermDict[1], indexTermDict[0])
				rankings = displayRanking(initial_rank, lower_to_upper_i, labeled_dict, flavor_dict, session['searchBy'])[:50]
				session['rankings'] = rankings
			session['xable'] = xable


	# user wants to add the item they queried to the cocktail
	if addQueryToCocktail:
		cocktail = session.pop('cocktail', [])
		# TODO: there are some queries not in lower_to_upper_i b/c apostrophes => fix this
		if (addQueryToCocktail in lower_to_upper_i):
			addSearchedIngredient = lower_to_upper_i[addQueryToCocktail]
		else:
			addSearchedIngredient = addQueryToCocktail
		if (addSearchedIngredient not in cocktail) and (addSearchedIngredient != None):
			cocktail.append(addSearchedIngredient)
		session['cocktail'] = cocktail

	# user wants to add item to cocktail
	if (addToCocktail != None):
		if (addToCocktail not in session['cocktail']):
			cocktail = session.pop('cocktail', [])
			cocktail.append(addToCocktail)
			session['cocktail'] = cocktail
		# if (addToCocktail not in session['xable']):
		# 	xable = session.pop('xable', [])
		# 	xable.append(addToCocktail)
		# 	session['xable'] = xable

	# user wants to clear the cocktail list
	if clearCocktail:
		session['cocktail'] = []

	# user wants to remove an item from the cocktail list
	if removeFromCocktail:
		cocktail = session.pop('cocktail', [])
		if (removeFromCocktail in cocktail):
			cocktail.remove(removeFromCocktail)
		session['cocktail'] = cocktail

	# user is done building recipe
	if done:
		ingred_query = queriesForCocktail(session['cocktail'])
		cocktail_flavor = createCocktailFlavor(ingred_query, flavor_dict)
		cocktail_list = makeCocktailRanks(ingred_query, makeJaccard, recipe_dict)
		return render_template('cocktails.html', cocktail=session['cocktail'], cocktail_search=cocktail_list,\
			cocktail_flavor = cocktail_flavor)


	# takes the user to the about page
	if about:
		return render_template('about.html', name=project_name, netid=net_id)

	# print("--- %s seconds ---" % (time.time() - start_time))
	# removed searched=ingredients for testing purposes
	return render_template('search.html', name=project_name, netid=net_id, \
		complete_ingredients=json.dumps(auto_ingredients_list), \
			output_message=session['output_message'], data=session['rankings'], \
			cocktail=session['cocktail'], search_by=session['searchBy'], xable=session['xable'], searched=ingredients)
