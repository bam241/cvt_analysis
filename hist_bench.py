import numpy as np
import pandas as pd


def get_data(file):
    countries=np.genfromtxt(file, delimiter="\t", usecols=0, dtype=str)
    raw_data=np.genfromtxt(file, delimiter="\t", usecols=(1,2,3,4,5,6,7,8), skip_header=1)

    # now get column names (how is there not a better way?)
    tmp=np.genfromtxt(file, delimiter="\t", usecols=(1,2,3,4,5,6,7,8),names=True)
    col_names=tmp.dtype.names

    return countries, col_names, raw_data


def calc_pursuit(raw_data, weights):
    final_vals = []
    weighted_factors = weights*raw_data
    for i in range(raw_data.shape[0]):
        val = weighted_factors[i].sum()
        final_vals.append(round(val,4))
    return final_vals

def power_law(x_points, a, B=1):
    # format B*x^a
    y_vals = np.power(x_points, a)
    y_vals = y_vals*B
    return y_vals

def frac_hist(spec_ys, tot_ys, bin_size=1):
    tot_hist, z = np.histogram(tot_ys, bin_size)
    spec_hist, z = np.histogram(spec_ys, bin_size)
    frac_hist, z = spec_hist.astype(float)/tot_hist.astype(float)

    return tot_hist, fract_hist


# Calculate the line that best fits the data using weighted least squares
# Assumes the ideal solution is of the form y = A + Bx
# (Input must be numpy arrays)
def ls_fit(xs, ys, ws=None):

    if ws is None:
        n_pts = xs.shape
        weights = np.ones(n_pts,)
    else:
        weights = ws
    
    print "weights ", weights 
    x_sq = weights*np.power(xs,2)
    sum_x_sq = x_sq.sum()
    sum_x = (weights*xs).sum()
    xy = weights*xs*ys
    sum_xy = xy.sum()
    sum_y = (weights*ys).sum()
    sum_w = weights.sum()

    denom = sum_w*sum_x_sq - np.power(sum_x, 2)
    A_num = sum_x_sq*sum_y - sum_x*sum_xy
    B_num = sum_w*sum_xy - sum_x*sum_y

    A = A_num/denom
    B = B_num/denom

    return A, B

    
#def rms_fit_quality(data, model):