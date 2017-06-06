#!/opt/local/bin/python3

### reverse forms, to check whether prefixing vs suffixing language behave differently

import regex as re
from glob import glob
import pandas as pd

def reverse(col):
    if ' ' in data[col][0]:
        data[col] = data[col].str.split(' ')
        data[col] = [' '.join(reversed(f)) for f in data[col]]
    else:
        data[col] = [re.findall(r'\X', f) for f in data[col]]
        data[col] = [''.join(reversed(f)) for f in data[col]]

for name in glob('data/*.dat.gz'):
    data = pd.read_csv(name, sep='\t', names=['form', 'lexeme', 'features', 'lemma'])
    if not data['lemma'].isnull().any():
        reverse('form')
        reverse('lemma')
        out_name = re.sub(r'\.dat\.gz', '-rev.dat.gz',  name)
        data.to_csv(out_name, compression='gzip', index=False, header=False, sep='\t',
                    columns=['form', 'lexeme', 'features', 'lemma'])

