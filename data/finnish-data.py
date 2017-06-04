#!/opt/local/bin/python3

## create Finnish noun/adj form database

import requests, tarfile, io, re
import pandas as pd

URL1 = 'https://korp.csc.fi/suomen-sanomalehtikielen-taajuussanasto-B9996.txt'
URL2 = 'http://www.cs.utexas.edu/~gdurrett/data/wiktionary-morphology-1.1.tgz'

words = set()
r = requests.get(URL1)
for line in r.content.splitlines():
    match = re.search(r'0,[0-9]+ ([^ ]+) ', line.decode('utf-8'))
    if match:
        words.add(match.group(1))

r = requests.get(URL2)
z = tarfile.open(fileobj=io.BytesIO(r.content), mode='r')
data = pd.read_csv(io.BytesIO(z.extractfile('wiktionary-morphology-1.1/inflections_fi_nounadj.csv').read()),
                    header=None, names=['form','lexeme','features'])
data['form'] = data['form'].str.lower()
data['lexeme'] = data['lexeme'].str.lower()
data = data[data['lexeme'].isin(words)]
data['lemma'] = data['lexeme']

data.to_csv('finnish.dat.gz', index=False, header=False,
            sep='\t', compression='gzip', columns=['form', 'lexeme', 'features', 'lemma'])
