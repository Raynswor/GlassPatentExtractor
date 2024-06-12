

from io import StringIO
import re
import ssl

import certifi
from kieta_data_objs import Document, Area, BoundingBox

from kieta_modules import Module
from kieta_modules import util

from typing import List, Dict, Optional, Tuple

import collections
import requests

import urllib3
import pandas as pd

from dateutil.parser import parse


class MetadataExtractor(Module):
    _MODULE_TYPE = 'MetadataExtractor'

    def __init__(self, stage: int, parameters: Optional[Dict] = {}, debug_mode: bool = True) -> None:
        super().__init__(stage, parameters, debug_mode)
        self.lines = parameters.get('lines', 'Line')
        self.strings = parameters.get('strings', 'String')

        retries = urllib3.Retry(connect=2, read=1, redirect=5)
        self.http = urllib3.PoolManager(retries=retries, cert_reqs='CERT_NONE', assert_hostname=False)

        # self.us_url = "https://assignment-api.uspto.gov/patent/lookup?filter=PatentNumber&query="
        # self.eu_url = "https://data.epo.org/linked-data/data/publication/"

        self.url = "https://depatisnet.dpma.de/DepatisNet/depatisnet?action=bibdat&docid="

    def execute(self, inpt: Document):
        candidates = list()
        # check if document oid is a valid patent number
        res = re.search(r'(?:[A-Z]{2})?\s?((?:\d+)[ ,/]?)+\s?(?:[A-Z]\d)?', inpt.oid)
        if res and len(res[0])> 5:
            candidates.append(res[0])
        else:
            for line in inpt.get_area_type(self.lines):
                s = ' '.join(inpt.get_area_data_value(line))
                
                res = re.search(r'(?:[A-Z]{2})?\s?((?:\d+)[ ,/]?)+\s?(?:[A-Z]\d)?', s)
                if res and len(res[0])> 5:
                    candidates.append(res[0].replace(' ', '').replace(',', ''))
        # majority count
        found_number = None
        if len(candidates) > 0:
            res = collections.Counter(candidates)
            self.debug_msg(res)
            found_number = res.most_common(1)[0][0]
        else:
            return None
        
        found_number = re.sub(r'[ ,/]', '', found_number)
        prefix = re.search(r'^[A-Z]{2}', found_number)
        if not prefix:
            prefix = "US"
        else :
            prefix = prefix[0]

        number = re.search(r'\d+', found_number)[0]
        suffix = re.search(r'[A-Z]\d$', found_number)
        if not suffix:
            # try several suffixes
            suffix = ['A1', 'A1', 'B1', 'B2']
        else:
            suffix = [suffix[0]]
        
        # if "US" in prefix:
        #     response = self.http.request('GET', self.us_url+number).json()
        # elif "EP" in prefix:
        #     response = self.http.request('GET', self.eu_url+f"EP/{number}/{suffix if suffix else 'A1'}/-.json").json()
        for s in suffix:
            self.info_msg(f"Trying {prefix}{number}{s}")

            response = self.http.request('GET', self.url+f"{prefix}{number}{s}").data

            response = response.decode('utf-8', 'ignore')
            # find <table
            response = re.search(r'<table.*</table>', response, re.DOTALL)
            if not response:
                continue
            response = response[0]

            # parse response, html table with 'INID' "Kriterium" "Feld" "Inhalt" as data-th 
            # make as dict
            try:
                dfs = pd.read_html(StringIO(response), header=0)[0]
            except ValueError as e:
                print(e)
                continue
            # iterate over dfs
            for ix, row in dfs.iterrows():
                cont = str(row['INID'])
                content = str(row['Inhalt'])
                if '54' in cont:
                    inpt.metadata['title'] = re.sub(r'^(?:\[[A-Z]{2}\])\s', '', content)
                if '72' in cont:
                    inpt.metadata['inventors'] = content
                if '71' in cont:
                    inpt.metadata['applicant'] = content
                if '73' in cont:
                    inpt.metadata['assignee'] = content
                if '43' in cont or '45' in cont or 'Veröffentlichungsdatum' in row['Kriterium']:
                    try:
                        parsed = parse(row['Inhalt'], fuzzy=True)
                    except:
                        continue
                    inpt.metadata['publication_date'] = parsed.strftime('%Y-%m-%d')
                if '51' in cont:
                    if ')' in content:
                        for t in content.split(')'):
                            t = t.strip()
                            if not t:
                                continue
                            try:
                                inpt.metadata['ipc'].append(t+')')
                            except KeyError:
                                inpt.metadata['ipc'] = [t+')']
                    elif '  ' in content:
                        for t in content.split('  '):
                            t = t.strip()
                            if not t:
                                continue
                            try:
                                inpt.metadata['ipc'].append(t)
                            except KeyError:
                                inpt.metadata['ipc'] = [t]
                    else:
                        try:
                            inpt.metadata['ipc'].append(content)
                        except KeyError:
                            inpt.metadata['ipc'] = [content]
                    # remove nan and duplicates
                    inpt.metadata['ipc'] = list(set([x for x in inpt.metadata['ipc'] if x != 'nan']))

                if '57' in cont:
                    inpt.metadata['abstract'] = content
                
                # if nan
            if inpt.metadata.get('publication_date', 'nan') == 'nan' or inpt.metadata.get('title', 'nan') == 'nan':
                continue
            else:
                inpt.metadata['patent_number'] = prefix+number+s
                break
        return inpt
