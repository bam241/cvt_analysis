#!/usr/bin/env python

import glob
import cymetric as cym

def make_list(path):
    file_list = glob.glob(path + "*.sqlite")
    for f in file_list:
         db = cym.dbopen(f)
         frame = cym.eval('WeaponProgress', db,
                          conds=['Decision', '==', 1])
         