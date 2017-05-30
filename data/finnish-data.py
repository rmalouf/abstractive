#!/opt/local/bin/python3

## create Finnish noun/adj form database

import requests, tarfile, io
import pandas as pd

URL = 'http://people.eecs.berkeley.edu/~gdurrett/data/wiktionary-morphology-1.1.tgz'

r = requests.get(URL)
z = tarfile.open(fileobj=io.BytesIO(r.content), mode='r')

data = pd.read_csv(io.BytesIO(z.extractfile('wiktionary-morphology-1.1/inflections_fi_nounadj.csv').read()), 
                    header=None, names=['form','lexeme','features'])

data.to_csv('finnish.dat.gz', index=False, header=False, 
            sep='\t', compression='gzip')
