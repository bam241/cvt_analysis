import numpy as np
import pandas as pd

# wrapper to import Cyclist data and be back-compatible
def import_data(filename, which_time) :
    return import_data_cyclist(filename, whichtime)

# imports inventory data from Cyclist output
# (having plotted Time vs InventoryQty, filtered by Prototype)
#  Time, Quantity, Prototype
def import_data_cyclist(filename, which_time) :
    raw_data = pd.read_csv(filename)

    LEU = raw_data[raw_data['Prototype'] == "LEU"]
    delta_LEU = raw_data[raw_data['Prototype'] == "delta_LEU"]
    covert_HEU = raw_data[raw_data['Prototype'] == "covert_HEU"]

    try:
        if which_time == "LEU":
            time = LEU['Time']
        elif which_time == "delta_LEU":
            time = delta_LEU["Time"]
        elif which_time == "covert_HEU":
            time = covert_HEU["Time"]
    except TypeError:
        print("Types are: LEU, delta_LEU, covert_HEU")
    
    LEU_tp =(LEU[' Quantity']- LEU[' Quantity'].shift(1))
    delta_LEU_tp =  delta_LEU[' Quantity']- delta_LEU[' Quantity'].shift(1)
    covert_HEU_tp =  covert_HEU[' Quantity']- covert_HEU[' Quantity'].shift(1)
    
    return LEU_tp, delta_LEU_tp, covert_HEU_tp, time

# import data from Cyan inventory format
# (2 column Time, Qty, separated by 1+ whitespace)
def import_data_cyan(filename) :
    raw_data = pd.read_csv(filename,sep='\s+')
    time = raw_data['Time']
    tp = raw_data['Quantity'] - raw_data['Quantity'].shift(1)

    return time, tp


# Convert Quantity into thruput, then take FFT of output LEU
def calc_fft(filename) :
    
    LEU_tp, delta_LEU_tp, covert_HEU_tp, time = import_data(filename,'LEU')
    n = (LEU_tp.size-1)
    fourier = np.fft.rfft(LEU_tp[1:])
    ps = np.abs(fourier)**2
    freqs = np.fft.rfftfreq(n, d=1)
    idx = np.argsort(freqs)
    
    return ps, freqs, idx


# Returns FFT of covert HEU signal
def fft_heu(filename) :
    LEU_tp, delta_LEU_tp, covert_HEU_tp = import_data(filename, 'covert_HEU')
    n = (covert_HEU_tp.size-1)
    fourier = np.fft.rfft(covert_HEU_tp[1:])
    ps = np.abs(fourier)**2
    freqs = np.fft.rfftfreq(n, d=1)
    idx = np.argsort(freqs)
    
    return ps, freqs, idx


def trunc_flt(d,precision = 3) :
    before_dec, after_dec = str(d).split('.')
    res = float('.'.join((before_dec, after_dec[0:precision])))

    return str(res)

# Convert quantity shipped into shipping frequency, with sub-timestep
# granularity
# Input:
#   filename: CYAN .dat inventory file, including absolute path
#   ship_increment: quantity that constitutes one shipment (ie 100kg)
# Return:
#   ship_freq: list of time elapsed since last shipment
#   qty_sum: remaining unshipped material at end of simulation
#   csvfile: if defined then write the ship_freq to a csv file
#
def ship_freq(filename, ship_increment, csvfile='no') :
    t_last = 0.
    qty_sum = 0.
    ship_intervals = []

    time, tp= import_data_cyan(filename)

    for i in range(len(tp)):  
        if i != 0: 
            qty_sum += tp[i]
            if (qty_sum >= ship_increment):
                delta_qty = qty_sum - ship_increment
                frac = 1 - (delta_qty/tp[i])
                delta_t_report = (time[i-1] + frac) - t_last
                ship_intervals.append(delta_t_report)
                t_last = time[i-1] + frac
                qty_sum = delta_qty
    
    if csvfile != 'no':
        import csv
        with open(csvfile, 'wb') as output:
            cw = csv.writer(output, delimiter= '\n')
            for i in ship_intervals:
                cw.writerow([i])

    return ship_intervals, qty_sum


# Convert quantity into unique shipments
# Input:
#   filename: CYAN .dat inventory file, including absolute path
#   ship_increment: quantity that constitutes one shipment (ie 100kg)
# Return:
#   ship_freq: list of time elapsed since last shipment
#   qty_sum: remaining unshipped material at end of simulation
#   csvfile: if defined then write the ship_freq to a csv file
#
def ship_times(filename, ship_increment, csvfile='no') :
    t_last = 0.
    qty_sum = 0.
    t_ship = []

    time, tp= import_data_cyan(filename)

    for i in range(len(tp)):  
        if i != 0: 
            qty_sum += tp[i]
            if (qty_sum >= ship_increment):
                delta_qty = qty_sum - ship_increment
                frac = 1 - (delta_qty/tp[i])
                t_last = time[i-1] + frac
                t_ship.append(t_last)
                qty_sum = delta_qty
    
    if csvfile != 'no':
        import csv
        with open(csvfile, 'wb') as output:
            cw = csv.writer(output, delimiter= '\n')
            for i in t_ship:
                cw.writerow([i])

    return t_ship, qty_sum
