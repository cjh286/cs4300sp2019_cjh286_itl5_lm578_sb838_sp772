# import requests, bs4, sys before running
# enter link as argument in this format: python whiskeyscraper.py link

import requests # to request webpage
import bs4 # beautiful soup
import sys

drinklink = sys.argv[1]

res = requests.get(drinklink)
soup = bs4.BeautifulSoup(res.text, 'html.parser')

# finesse into the right div
body = soup.find('body')
outmostdiv = body.find('div', {"id": "siteWrapper"})
nextlayer = outmostdiv.find('div', {"id": "content"})
reviewcontainer = nextlayer.find('div', {"class": "wrapper product-reviews"}).find('div', {"class": "container"})
reviewslist = reviewcontainer.find('div', {"class": "reviews"}).find("ul", {"class": "reviews-list"})


reviewstring = ""

for li in reviewslist.findAll("li"):
    reviews = (li.find("p").getText().lower()).split()
    reviewstring += ' '.join(reviews)

print(reviewstring)
