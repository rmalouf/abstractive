#!/usr/bin/env python3

### prepare Russian noun forms from Brown and Hippisley
###

import requests, io, re
import pandas as pd

URL = 'http://networkmorphology.as.uky.edu/sites/default/files/ch23_rusnoms.dmp'
r = requests.get(URL)

data = [ ]
for line in r.content.decode().splitlines():
    parse = re.match(r'^([A-Za-z]+):<mor ([a-z ]+)> = (.*)\.', line)
    if parse:
        lexeme = parse.group(1).lower()
        feats = parse.group(2)
        form = parse.group(3)
        if form != 'undefined':
            form = form.replace('-&', '@')
            form = form.replace('^', '')
            form = form.replace('_', '')
            form = form.replace(' ', '')
            form = form.replace('(u)', '')
            form = form.replace('"', '')
            if '@' in form:
                # shift stress marker onto vowel
                form = re.sub(r'([aeiou])([^aeiou]*)@', r'\1@\2', form)
            data.append({'form':form, 'lexeme':lexeme, 'features':feats})

data = pd.DataFrame(data, columns=['form', 'lexeme', 'features'])
data.to_csv('russian.dat.gz', compression='gzip', header=None, sep='\t', 
                columns=['form', 'lexeme', 'features'], index=False)

            
                
