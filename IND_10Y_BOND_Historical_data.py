# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 06:55:03 2019

@author: khushal
"""

from requests import get
from bs4 import BeautifulSoup
from bs4 import SoupStrainer
import sys

'''
if sys.version_info >= (3,):
    import urllib.request as urllib2

else:
    import urllib2
'''

hdr = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Referer': 'https://cssspritegenerator.com',
    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
    'Accept-Encoding': 'none',
    'Accept-Language': 'en-US,en;q=0.8',
    'Connection': 'keep-alive'}

historical_bond_data_url = "https://in.investing.com/rates-bonds/india-10-year-bond-yield-historical-data"

res= get(historical_bond_data_url,headers=hdr)
print ("Querying = " + historical_bond_data_url)
if res.status_code == 200:
    soup=BeautifulSoup(res.text, "lxml")
    table_body=soup.find('tbody')
    rows = table_body.find_all('tr')
    #print(rows)
    for row in rows:
        cols=row.find_all('td')
        cols=[x.text.strip() for x in cols]
        print(cols)

else:
    print("Error in connecting to ",historical_bond_data_url,"having status code as ",res.status_code)