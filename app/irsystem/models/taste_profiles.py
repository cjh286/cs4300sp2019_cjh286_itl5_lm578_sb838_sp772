from collections import defaultdict 
import csv
import re
import openpyxl

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

# read in excel data as set
tastes = []

book = openpyxl.load_workbook('../../../scraped_data/taste_words.xlsx')
sheet = book.get_sheet_by_name('Sheet1')
for i in range(1, sheet.max_row+1):
    tastes.append(sheet.cell(row=i, column=1).value)

tasteset = set(tastes)


drink_flavors_dict = {}

for drink in alc_dict.keys():
    reviewset = set(alc_dict.get(drink))
    overlapping_flavors = reviewset.intersection(tasteset)

    drink_flavors_dict[drink] = list(overlapping_flavors)

print(drink_flavors_dict['1000 Stories174 Zinfandel - 750ml Bottle'])