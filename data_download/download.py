from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup

import json
import os
import argparse

import requests
import urllib
import urllib3
from urllib3.exceptions import InsecureRequestWarning

import datetime
import time
import sys

urllib3.disable_warnings(InsecureRequestWarning)

searchword1 = 'vánoční stromeček'
# searchword1 = 'christmas tree'
searchurl = 'https://www.google.com/search?q=' + searchword1 + '&source=lnms&tbm=isch'
dirs = searchword1
maxcount = 1000

chromedriver = r"C:\Users\Tom\Downloads\chromedriver_win32 (1)\chromedriver.exe"

if not os.path.exists(dirs):
    os.mkdir(dirs)

def download_google_staticimages():

    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    #options.add_argument('--headless')

    try:
        browser = webdriver.Chrome(chromedriver, options=options)
    except Exception as e:
        print(f'No found chromedriver in this environment.')
        print(f'Install on your machine. exception: {e}')
        sys.exit()

    browser.set_window_size(1280, 1024)
    browser.get(searchurl)
    time.sleep(1)

    print(f'Getting you a lot of images. This may take a few moments...')

    element = browser.find_element_by_tag_name('body')

    for qq in range(5):
        # Scroll down
        #for i in range(30):
        for i in range(50):
            element.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.3)
    
        try:
            browser.find_element_by_id('smb').click()
            for i in range(50):
                element.send_keys(Keys.PAGE_DOWN)
                time.sleep(0.3)
        except:
            for i in range(10):
                element.send_keys(Keys.PAGE_DOWN)
                time.sleep(0.3)
    
        print(f'Reached end of page.')
        time.sleep(0.5)
        print(f'Retry')
        time.sleep(0.5)
    
        # Below is in japanese "show more result" sentences. Change this word to your lanaguage if you require.
        try:
            browser.find_element_by_xpath('//input[@value="Zobrazit další výsledky"]').click()
        except:
            break

   
    
    #page_source = elements[0].get_attribute('innerHTML')
    page_source = browser.page_source 

    soup = BeautifulSoup(page_source, "html.parser")
    images = soup.find_all('img')

    urls = []
    for image in images:
        try:
            url = image['data-src']
            if not url.find('https://'):
                urls.append(url)
        except:
            try:
                url = image['src']
                if not url.find('https://'):
                    urls.append(image['src'])
            except Exception as e:
                print(f'No found image sources.')
                print(e)

    count = 0
    if urls:
        for url in urls:
            try:
                res = requests.get(url, verify=False, stream=True)
                rawdata = res.raw.read()
                
                with open(os.path.join(dirs, 'img_' + str(count)), 'wb') as f:
                    f.write(rawdata)
                    count += 1
                    
               
                    
            except Exception as e:
                print('Failed to write rawdata.')
                print(e)
                
                
            if count >=maxcount:
                break

    browser.close()
    return count

# Main block
def main():
    t0 = time.time()
    count = download_google_staticimages()
    t1 = time.time()

    total_time = t1 - t0
    print(f'\n')
    print(f'Download completed. [Successful count = {count}].')
    print(f'Total time is {str(total_time)} seconds.')

if __name__ == '__main__':
    main()
    
    
    
    
    