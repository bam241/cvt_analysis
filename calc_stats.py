#!/usr/bin/env python

import glob
import cymetric as cym
import pandas as pd

# Make a list of time for first proliferation in each file in the directory
# make a list of all sqlite files in the directory
def make_list(path):
    file_list = glob.glob(path + "*.sqlite")
    return file_list

# Return a list of the column value at the first incident of proliferation
# for all simulations in the directory
def first_prolif_qty(path, qty, csv = None):
    file_list = make_list(path)
    qty_df=pd.DataFrame()
    for f in file_list:
         db = cym.dbopen(f)
         weapon_progress = cym.root_metric(name='WeaponProgress')
         evaluator = cym.Evaluator(db)
         frame = evaluator.eval('WeaponProgress', conds=[('Decision', '==', 1)])
         qty_df=pd.concat([qty_df,frame[:1][qty]],ignore_index=True)
         
    if (csv != None):
        qty_df.to_csv(path+csv, sep='\t')
        return
    else:
        return qty_df

# Return a list of the column value for every incident of proliferation, for
# all simulations in the directory
def all_prolif_qty(path, qty, csv = None):
    file_list = make_list(path)
    qty_df=pd.DataFrame()
    for f in file_list:
         db = cym.dbopen(f)
         weapon_progress = cym.root_metric(name='WeaponProgress')
         evaluator = cym.Evaluator(db)
         frame = evaluator.eval('WeaponProgress', conds=[('Decision', '==', 1)])
         qty_df=pd.concat([qty_df,frame[:][qty]],ignore_index=True)

    if (csv != None):
        qty_df.to_csv(path+csv, sep='\t')
        return
    else:
        return qty_df




         
                      
        