import requests
from bs4 import BeautifulSoup

def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract text from the website
    text = soup.get_text(separator=' ')
    
    # Split text into sentences or paragraphs (adjust as needed)
    documents = [para for para in text.split('\n') if para.strip()]
    
    return documents

# Scrape data from the website
website_url = "https://www.dsvv.ac.in"
documents = scrape_website(website_url)