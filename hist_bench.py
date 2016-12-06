import numpy as np
import pandas as pd


def get_data(file):
    n_header = 2
    countries=np.genfromtxt(file, delimiter="\t", usecols=0, dtype=str,
                            skip_header=n_header)
    raw_data=np.genfromtxt(file, delimiter="\t", usecols=(1,2,3,4,5,6,7,8),
                           skip_header=n_header)

    # now get column names (how is there not a better way?)
    tmp=np.genfromtxt(file, delimiter="\t", usecols=(1,2,3,4,5,6,7,8),
                      names=True, skip_header=n_header-1)
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

# Weapon states and their acquire date
def get_nws():
    nws = {}
    nws["China"] = 1964
    nws["France"] = 1960
    nws["India"] = 1988
    nws["Israel"] = 1969
    nws["North Korea"] = 2006
    nws["Pakistan"] = 1987
    nws["South Africa"] = 1979
    nws["United Kingdom"] = 1952
    nws["United States"] = 1945
    nws["USSR"] = 1949

    return nws

# States that pursued and their pursuit date
# (from google doc: Main Prolif Spreadsheet, Dec-5-2016)
def get_pursue():
    pursues = {}
    pursues["Argentina"] = 1978
    pursues["Australia"] = 1961
    pursues["Brazil"] = 1978
    pursues["China"] = 1955
    pursues["Egypt"] = 1965
    pursues["France"] = 1954
    pursues["India"] = 1964
    pursues["Iran"] = 1985
    pursues["Iraq"] = 1983
    pursues["Israel"] = 1960
    pursues["Libya"] = 1970
    pursues["North Korea"] = 1980
    pursues["South Korea"] = 1970
    pursues["Pakistan"] = 1972
    pursues["South Africa"] = 1974
    pursues["Syria"] = 2000
    pursues["United Kingdom"] = 1947
    pursues["United States"] = 1939
    pursues["USSR"] = 1945

    return pursues

# From a matched pair of numpy arrays containing countries and their pursuit scores,
# make a new array of the pursuit scores for countries that succeeded
def get_prolif_pe(countries, pes):
    prolif_pes = []
    prolif_st = []
    proliferants = get_prolif()
    for i in range(len(countries)):
        curr_state = countries[i]
        if curr_state in proliferants:
            prolif_pes.append(pes[i])
            prolif_st.append(curr_state)
    return(prolif_st, prolif_pes)

# From a matched pair of numpy arrays containing countries and their pursuit
# scores, make a new array of the scores for the subset of countries that
# actually chose to pursue and/or succeeded
def get_pes(all_countries, pes, status):
    pursue_pes = []
    pursue_st = []
    if (status == "Pursue"):
        states = get_pursue()
    elif (status == "Prolif"):
        states = get_nws()
    else:
        return "DO YOU WANT PURSUIT OR PROLIFERANTS?"
    for i in range(len(all_countries)):
        curr_state = all_countries[i]
        if curr_state in states:
            pursue_pes.append(pes[i])
            pursue_st.append(curr_state)
    return(pursue_st, pursue_pes)