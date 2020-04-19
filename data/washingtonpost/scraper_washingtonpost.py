import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
import argparse
import bs4
import json
from tqdm import tqdm

class scraper:
	def __init__(self):
		self.url = "https://www.washingtonpost.com/coronavirus/"

	def run(self, num, filename):
		driver = webdriver.Chrome(ChromeDriverManager().install())
		driver.get(self.url)
		driver.implicitly_wait(100)
		soup = BeautifulSoup(driver.page_source, 'html.parser')
		div = soup.find("div",{'data-chain-name':"virus-stream-2"})
		contents = div.find_all(class_="headline")

		while len(contents)<num:

			button = driver.find_elements_by_class_name("skin-button-load-more")
			try:
				button[0].click()
			except: 
				driver.implicitly_wait(2000)
				continue
			driver.implicitly_wait(2000)
			soup = BeautifulSoup(driver.page_source, 'html.parser')
			div = soup.find("div",{'data-chain-name':"virus-stream-2"})
			contents = div.find_all(class_="headline")	
			print("Getting ", len(contents), " Covid-19 news link from Washington Post")	

		try:
			driver.close()
		except:
			pass
		f = open(filename, 'w', encoding='utf-8')
		print("Start extracting content from links!")
		pos = 0
		for entry in tqdm(contents):
			link = entry.find('a')['href']
			try:
				obj = self.get_json(link)

			except:
				continue
			json.dump(obj, f) 
			f.write("\n")
			pos+=1
			
		print("Success ", pos, "/", len(contents))
		f.close()


	def get_json(self,link):
	    page = requests.get(link)
	    soup = BeautifulSoup(page.content, 'html.parser')
	    title = soup.find("h1",{'data-qa':"headline"}).contents[0]
	    div = soup.find("div",{'class':"remainder-content"})
	    paragraphs = div.find_all('p')
	    content = "\n".join([str(p.contents[0]) for p in paragraphs if type(p.contents[0]) == bs4.element.NavigableString])
	    author = [str(a.contents[0]) for a in soup.find_all('span',{"class": "author-name"})]
	    date = str(soup.find("div",{'class':"display-date"}).contents[0])
	    res = {
	        'title':title,
	        'time': date,
	        'author':author,
	        'link':link,
	        'content':content,
	                    }
	    return res
	# def get_json(self, link):





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--o', type=str, default="output.json")
    args = parser.parse_args()
    cp = scraper()
    cp.run(args.n, args.o)
