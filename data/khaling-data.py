#!/usr/bin/env python3

# Get Khaling noun forms from Khalex

# http://www.llf.cnrs.fr/sites/sandbox.linguist.univ-paris-diderot.fr/files/statiques/flexique/flexique.zip

import requests, tarfile, io
import pandas as pd

URL = 'https://gforge.inria.fr/frs/download.php/file/35119/khalex-0.0.2.mlex.tgz'

r = requests.get(URL)
z = tarfile.open(fileobj=io.BytesIO(r.content), mode='r:gz')

data = pd.read_csv(io.BytesIO(z.extractfile('khalex-0.0.2.mlex/khalex-0.0.2.mlex').read()), sep='\t', 
    comment=None, quoting=3, names=['form', 'pos', 'lexeme', 'features'])
data = data[data['pos']=='V']
del data['pos']

data['form'] = [w.lower() for w in data['form']]
data.to_csv('khaling.dat.gz', index=False, header=False, 
            sep='\t', compression='gzip')
