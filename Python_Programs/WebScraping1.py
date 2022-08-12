import bs4
import requests

res = requests.get("https://www.simplesite.com/")


print(type(res))
print(res.text)
soup = bs4.BeautifulSoup(res.text, 'lxml')

title = soup.select('title')
print(title[0].getText())

print("title is")
print(title[0].getText())

arr =  soup.select(".mw.headline")
for element in arr:
	print(element.text)