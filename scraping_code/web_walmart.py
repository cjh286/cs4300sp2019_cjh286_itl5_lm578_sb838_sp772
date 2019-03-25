import requests
import bs4

# write output to file
file = open("walmart_beverages.txt","w+", encoding = 'utf-8')

# find all recipe links across all pages
for pagenum in range(1,26):
    urlstring = 'https://www.walmart.com/browse/beverages/976759_976782?page=' + str(pagenum)
    res = requests.get(urlstring)
    
    soup = bs4.BeautifulSoup(res.text, 'html.parser') 

    # extract links
    # print(soup.findAll('a', {"class": "product-title-link line-clamp line-clamp-2"}))
    for drank in soup.findAll('a', class_='product-title-link line-clamp line-clamp-2'):
    	# print(tile.findAll('a')[0].get('href'))
        descriptor = drank['aria-label']
        file.write(descriptor+ '\n')
        

file.close()