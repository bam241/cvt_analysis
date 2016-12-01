import numpy as np
import pandas as pd


def import_data(file):
    countries=np.genfromtxt(file, delimiter="\t", usecols=0, dtype=str)
    raw_data=np.genfromtxt(file, delimiter="\t", usecols=(1,2,3,4,5,6,7,8,9,10), skip_header=2)

    # now get column names (how is there not a better way?)
    tmp=np.genfromtxt(file, delimiter="\t", usecols=(1,2,3,4,5,6,7,8,9,10),names=True)
    col_names=tmp.dtype.names

    return countries, col_names, raw_data


def calc_pursuit(raw_data, weights):
    final_vals = []
    weighted_factors = weights*raw_data
    for i in range(raw_data.shape[0]):
        val = weighted_factors[i].sum()/10.0
        final_vals.append(round(val,4))
    return final_vals

def power_law(x_points, a, B=1):
    # format B*x^a
    y_vals = np.power(x_points, a)
    y_vals = y_vals*B
    return y_vals


# Write an exponential least-squares fit routine?
# OR write a RMS calculator??
# Or both?
#def ls_fit(data, model):
    