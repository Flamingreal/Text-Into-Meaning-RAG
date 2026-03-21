import requests
from bs4 import BeautifulSoup
import re
import time
import os

def clean_wiki_text(url):
    try:
        # Set user agent to avoid being blocked
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get page title
        title_element = soup.find('h1')
        title = title_element.text if title_element else "Unknown Title"
        
        # Get main content area
        content = soup.find('div', {'class': 'mw-parser-output'})
        if not content:
            return ""
        
        # Remove references like [1] and edit links
        for sup in content.find_all('sup', {'class': 'reference'}):
            sup.decompose()
        for span in content.find_all('span', {'class': 'mw-editsection'}):
            span.decompose()
            
        paragraphs = content.find_all(['p', 'h2', 'h3', 'li'])
        
        clean_lines = [f"# {title}\n"]
        for p in paragraphs:
            text = p.get_text().strip()
            
            if text.startswith("Cookbook |") or "Cookbook Disambiguation Pages" in text:
                continue
            if "Incomplete recipes" in text or "deletion policy" in text or "meaningful content" in text:
                continue
                
            if len(text) > 5: 
                text = re.sub(r'\s+', ' ', text)
                clean_lines.append(text)
                
        return "\n\n".join(clean_lines)
    
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return ""
    
# List of Wikipedia and Wikibooks URLs to scrape
urls = [
    "https://en.wikipedia.org/wiki/List_of_Asian_cuisines#East_Asian_cuisine",
    "https://en.wikipedia.org/wiki/Chinese_cuisine",
    "https://en.wikipedia.org/wiki/Anhui_cuisine",
    "https://en.wikipedia.org/wiki/Cantonese_cuisine",
    "https://en.wikipedia.org/wiki/Fujian_cuisine",
    "https://en.wikipedia.org/wiki/Hunan_cuisine",
    "https://en.wikipedia.org/wiki/Jiangsu_cuisine",
    "https://en.wikipedia.org/wiki/Shandong_cuisine",
    "https://en.wikipedia.org/wiki/Sichuan_cuisine",
    "https://en.wikipedia.org/wiki/Zhejiang_cuisine",
    "https://en.wikipedia.org/wiki/Dim_sum",
    "https://en.wikipedia.org/wiki/Hot_pot",
    "https://en.wikipedia.org/wiki/Wine_in_China",
    "https://en.wikipedia.org/wiki/Char_siu",
    "https://en.wikipedia.org/wiki/Sichuan_peppercorn",
    "https://en.wikipedia.org/wiki/Huaiyang_cuisine",
    "https://en.wikipedia.org/wiki/Chinese_Islamic_cuisine",
    "https://en.wikipedia.org/wiki/Beijing_cuisine",
    "https://en.wikipedia.org/wiki/Chinese_aristocrat_cuisine",
    "https://en.wikipedia.org/wiki/Chinese_imperial_cuisine",
    "https://en.wikipedia.org/wiki/Liaoning_cuisine",
    "https://en.wikipedia.org/wiki/Chaozhou_cuisine",
    "https://en.wikipedia.org/wiki/Chiuchow_cuisine",
    "https://en.wikipedia.org/wiki/Guizhou_cuisine",
    "https://en.wikipedia.org/wiki/Hainan_cuisine",
    "https://en.wikipedia.org/wiki/Hakka_cuisine",
    "https://en.wikipedia.org/wiki/Henan_cuisine",
    "https://en.wikipedia.org/wiki/Hubei_cuisine",
    "https://en.wikipedia.org/wiki/Jiangxi_cuisine",
    "https://en.wikipedia.org/wiki/Manchu_cuisine",
    "https://en.wikipedia.org/wiki/Northeastern_Chinese_cuisine",
    "https://en.wikipedia.org/wiki/Shaanxi_cuisine",
    "https://en.wikipedia.org/wiki/Shanghai_cuisine",
    "https://en.wikipedia.org/wiki/Shanxi_cuisine",
    "https://en.wikipedia.org/wiki/Tianjin_cuisine",
    "https://en.wikipedia.org/wiki/Tibetan_cuisine",
    "https://en.wikipedia.org/wiki/Uyghur_cuisine",
    "https://en.wikipedia.org/wiki/Yunnan_cuisine",
    "https://en.wikipedia.org/wiki/Hong_Kong_cuisine",
    "https://en.wikipedia.org/wiki/Fish_balls",
    "https://en.wikipedia.org/wiki/Wonton_noodle",
    "https://en.wikipedia.org/wiki/Egg_waffle",
    "https://en.wikipedia.org/wiki/Japanese_cuisine",
    "https://en.wikipedia.org/wiki/Japanese_regional_cuisine",
    "https://en.wikipedia.org/wiki/Kaiseki",
    "https://en.wikipedia.org/wiki/Sushi",
    "https://en.wikipedia.org/wiki/Sashimi",
    "https://en.wikipedia.org/wiki/Japanese_wine",
    "https://en.wikipedia.org/wiki/Okinawan_cuisine",
    "https://en.wikipedia.org/wiki/Awamori",
    "https://en.wikipedia.org/wiki/Nagoya_cuisine",
    "https://en.wikipedia.org/wiki/Ainu_cuisine",
    "https://en.wikipedia.org/wiki/Korean_cuisine",
    "https://en.wikipedia.org/wiki/Banchan",
    "https://en.wikipedia.org/wiki/Kimchi",
    "https://en.wikipedia.org/wiki/Doenjang",
    "https://en.wikipedia.org/wiki/Korean_soy_sauce",
    "https://en.wikipedia.org/wiki/Gochujang",
    "https://en.wikipedia.org/wiki/Korean_regional_cuisine",
    "https://en.wikipedia.org/wiki/Korean_barbecue",
    "https://en.wikipedia.org/wiki/Soju",
    "https://en.wikipedia.org/wiki/Makgeolli",
    "https://en.wikipedia.org/wiki/Korean_royal_court_cuisine",
    "https://en.wikipedia.org/wiki/Korean_temple_cuisine",
    "https://en.wikipedia.org/wiki/North_Korean_cuisine",
    "https://en.wikipedia.org/wiki/South_Korean_cuisine",
    "https://en.wikipedia.org/wiki/Mongolian_cuisine",
    "https://en.wikipedia.org/wiki/Taiwanese_cuisine",
    "https://en.wikipedia.org/wiki/Khorkhog",
    "https://en.wikipedia.org/wiki/Japanese_Chinese_cuisine",
    "https://en.wikipedia.org/wiki/Shippoku",
    "https://en.wikipedia.org/wiki/Itameshi",
    "https://en.wikipedia.org/wiki/Yoshoku",
    "https://en.wikipedia.org/wiki/Korean_Chinese_cuisine",
    "https://en.wikipedia.org/wiki/Hot_and_sour_noodles",
    "https://en.wikipedia.org/wiki/Xiaolongbao",
    "https://en.wikipedia.org/wiki/Xuzhou_cuisine",
    "https://en.wikipedia.org/wiki/Haipai_cuisine",
    "https://en.wikipedia.org/wiki/Qinghai_cuisine",
    "https://en.wikipedia.org/wiki/Guilin_cuisine",
    "https://en.wikipedia.org/wiki/Putian_cuisine",
    "https://en.wikipedia.org/wiki/Teochew_cuisine",
    "https://en.wikipedia.org/wiki/Ou_cuisine",
    "https://en.wikipedia.org/wiki/Kachin_cuisine",
    "https://en.wikipedia.org/wiki/Hmong_cuisine",
    "https://en.wikipedia.org/wiki/Taoist_diet",
    "https://en.wikipedia.org/wiki/History_of_Chinese_cuisine",
    "https://en.wikipedia.org/wiki/History_of_Japanese_cuisine",
    "https://en.wikibooks.org/wiki/Cookbook:Broccoli_Stir_Fry",
    "https://en.wikibooks.org/wiki/Cookbook:Cream_Cheese_Wontons",
    "https://en.wikibooks.org/wiki/Cookbook:Fried_Rice",
    "https://en.wikibooks.org/wiki/Cookbook:Onigiri",
    "https://en.wikibooks.org/wiki/Cookbook:Spicy_Miso_Udon",
    "https://en.wikibooks.org/wiki/Cookbook:Wonton_Soup"
]

all_text = ""
print("Starting to scrape East Asian cuisine corpus...")

for url in urls:
    print(f"Fetching: {url}")
    scraped_text = clean_wiki_text(url)
    if scraped_text:
        all_text += scraped_text + "\n\n"
    # Delay to prevent IP blocking
    time.sleep(1.5)

# Save to data directory if it exists, otherwise use current directory
output_dir = "../data" if os.path.exists("../data") else "."
output_filename = os.path.join(output_dir, "East_Asian_Corpus_Massive.txt")

with open(output_filename, "w", encoding="utf-8") as f:
    f.write(all_text)

print(f"\nDone! Saved to: {output_filename}")
print(f"Total characters: {len(all_text)}")