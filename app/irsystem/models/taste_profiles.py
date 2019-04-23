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

def create_flavor_dict():
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
        list_to_grab_negation = alc_dict[drink]
        # no_terms = [w for i,w in list_to_grab_negation if i and (list_to_grab_negation[i-1] in ["no"])]
        # not_terms = [w for i,w in list_to_grab_negation if i and (list_to_grab_negation[i-1] in ["not"])]
        # never_terms = [w for i,w in list_to_grab_negation if i and (list_to_grab_negation[i-1] in ["never"])]

        to_remove_from_set = []
        get_negative_phrase = {}
        for w in range(len(list_to_grab_negation)-1):
            if list_to_grab_negation[w] == 'no' or list_to_grab_negation[w] == 'not' or list_to_grab_negation[w] == 'never':
                to_remove_from_set.append(list_to_grab_negation[w+1])
                get_negative_phrase[list_to_grab_negation[w+1]] = list_to_grab_negation[w] + " " + list_to_grab_negation[w+1]

        negative_terms = set(to_remove_from_set)

        reviewset = set(alc_dict.get(drink))
        for word in negative_terms:
            reviewset.remove(word)

        overlapping_flavors = reviewset.intersection(tasteset)


        drink_flavors_dict[drink] = list(overlapping_flavors) #list of positive descriptive words

        #check if any of the negative terms like 'flavorful' from 'not flavorful' are in the tasteset
        overlapping_negative_flavors = negative_terms.intersection(tasteset)
        for word in overlapping_negative_flavors: 
            drink_flavors_dict[drink].append(get_negative_phrase[word]) #word is 'flavorful' so add 'not flavorful' to the dictionary
    return drink_flavors_dict

def main():
    flavor_dict = create_flavor_dict()
    print(flavor_dict)

if __name__ == "__main__":
    main()