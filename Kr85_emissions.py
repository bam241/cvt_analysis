#!/usr/bin/env python


import cymetric as cym
import pandas as pd
import numpy as np
from cymetric import root_metric


#frame = cym.eval('Transactions', db, conds=[('Commodity', '==', 'spent_uox', 'RecevierId' > 14)])
#commods = frame[:]['Commodity'].unique()
#filtered_frame = cym.eval('Materials', db, conds=[('NucId', '==', 922350000)])


def FacilityInput(dbname):
    db = cym.dbopen(dbname)

    frame = cym.eval('TransactionQuantity', db)

    ids = frame[:]['ReceiverId'].unique()
    ids.sort()

    it = 0
    for cur_id in ids:
        cur_frame = cym.eval('TransactionQuantity', db, conds=[('ReceiverId', '==', cur_id)])
        if it == 0:
            big_frame = cur_frame[:]['TimeCreated']

        big_frame = pd.concat([big_frame, cur_frame[:]['Quantity']], axis=1)
        it += 1

    str_fac = map(str,ids)
    big_frame.columns = ['TimeCreated']+str_fac
    return big_frame




dir = '/Users/mbmcgarry/git/data_analysis/data/testing/random_sep/'
dbfile = dir+'norm_dist.sqlite'

velocity = 3.0 # m/s
size = 4
source = [0,1]

pos_matrix = np.ndarray((size,size))*0
pos_matrix[source[0]][source[1]] = 1

inventory_frame = FacilityInput(dbfile)
n_times = inventory_frame['TimeCreated'].size

emissions = np.ndarray((n_times,size,size))*0

