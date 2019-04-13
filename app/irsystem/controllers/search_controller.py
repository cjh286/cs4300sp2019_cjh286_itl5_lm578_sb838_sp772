from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
from app.irsystem.models.search import *
import os

project_name = "Drink Up!"
net_id = "cjh286, itl5, lm578, sb838, sp772"

@irsystem.route('/', methods=['GET'])
def search():
	ingredients = request.args.get('ingredients')

	if not ingredients:
		rankings = []
		output_message = "Nothing"
		
	else:
		output_message = "Your Search: " + ingredients
		query = [ingredients]
		drinks_list, all_ingredients_list, recipe_dict = build_recipe_dict()
		ingredients_dict = build_ingredients_dict(recipe_dict)
		indexTermDict = indexDict(all_ingredients_list)
		co_oc = makeCoOccurrence(recipe_dict, len(all_ingredients_list), indexTermDict[1])
		rankings = complementRanking(query, co_oc, indexTermDict[1], indexTermDict[0])

	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=rankings)



