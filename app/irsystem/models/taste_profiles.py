from collections import defaultdict 
import csv
import re

def tokenize(text): 
    """ Taken from class code A1
    Returns a list of words that make up the text.
    
    Note: for simplicity, lowercase everything.
    Requirement: Use Regex to satisfy this function
    
    Params: {text: String}
    Returns: List 
    """
    # YOUR CODE HERE
    x = re.findall(r'\b([a-zA-Z]+)\b',text)
    for y in range(len(x)):
        char = x[y].lower()
        x[y] = char
    return x




alc_dict = {} #store alc names and review text
with open('../../../scraped_data/alc reviews.csv', encoding="utf8") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
    	if not alc_dict.get(row[13]):
    		alc_dict[row[13]] = tokenize(row[23])
    	else:
    		alc_dict[row[13]].extend(tokenize(row[23]))



print(alc_dict['1000 Stories174 Zinfandel - 750ml Bottle'])
print(len(alc_dict.keys()))
