#!/usr/bin/env python


import cymetric as cym
import pandas as pd
import numpy as np
import math
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



def calc_emissions(db):
 
    velocity = 0.6 # m/s
    rows = 3
    cols = rows
    src = [0,1]   # [row,col]
    n_times = 6
    curr_emiss = abs(np.ndarray(n_times)*0)
    curr_emiss[0] = 100
    curr_emiss[2] = 50
    curr_emiss[3] = 50
    
    pos_matrix = abs(np.ndarray((rows,cols))*0)
    pos_matrix[src[0]][src[1]] = 1
    
    inventory_frame = FacilityInput(dbfile)
    #n_times = inventory_frame['TimeCreated'].size

    emissions = abs(np.ndarray((n_times,rows,cols))*0)
    
    for r in range(rows):
        for c in range(cols):
            dist = math.sqrt((r - src[0])**2 + (c - src[1])**2)
            pos_matrix[r][c] = dist
    fall_off_matrix = 1/(4*math.pi*(1+pos_matrix)**2)
    time_delay_matrix = pos_matrix/velocity

    for t in range(n_times):
        if curr_emiss[t] !=0 :
            eff_emiss = curr_emiss[t]*fall_off_matrix
            for r in range(rows):
                for c in range(cols):
                    eff_time_delay = time_delay_matrix[r][c]
                    if (t + eff_time_delay) <= n_times:  # change for real program
                        emissions[t+eff_time_delay][r][c] += eff_emiss[r][c]
    return emissions




import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.animation as animation

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

dir = '/Users/mbmcgarry/git/data_analysis/data/testing/random_sep/'
dbfile = dir+'norm_dist.sqlite'

emissions = calc_emissions(dbfile)

#time = 0
n_times = emissions.shape[0]
xmin = 0
ymin = 0
xmax = emissions.shape[1]
ymax = emissions.shape[2]
edge = 0.5
x = np.arange(xmax)
y = np.arange(ymax)
X, Y = np.meshgrid(x, y)
#Z = emissions[time][:][:]


# Create a simple contour plot with labels using default colors.  The
# inline argument to clabel will control whether the labels are draw
# over the line segments of the contour, removing the lines beneath
# the label
fig = plt.figure()
plt.xlabel('x location')
plt.ylabel('y location')
min_em = np.amin(emissions, axis=2).min()
max_em = np.amax(emissions, axis=2).max()
n_lev = (max_em - min_em)/25

#levels = np.arange(-1.2, 1.6, 0.2)
levels = np.arange(min_em, max_em, n_lev)

def animate(i):
    Z = emissions[i][:][:]
    cont = plt.imshow(Z, interpolation='bilinear', origin='lower',
                extent=(xmin-edge,xmax+edge,ymin-edge,ymax+edge))
    plt.title("Kr85 emissions, t="+ str(i))
    # We can still add a colorbar for the image, too.
    return cont

anim=1
if anim == 1:
    anim = animation.FuncAnimation(fig, animate, frames=n_times, repeat_delay=2000, interval=500, blit=False) 
else:
   cont = plt.imshow(emissions[5][:][:], interpolation='bilinear', origin='lower',
                extent=(xmin-edge,xmax+edge,ymin-edge,ymax+edge))
   CB = plt.colorbar(cont, shrink=0.8, extend='both')



#CBI = plt.colorbar(anim, shrink=0.8)
 

#plt.figure()
#im = plt.imshow(Z, interpolation='bilinear', origin='lower',
#                extent=(xmin-edge,xmax+edge,ymin-edge,ymax+edge))
#CS = plt.contour(Z, levels,
#                 origin='lower',
#                 linewidths=2,
#                 extent=(xmin,xmax,ymin,ymax))



# make a colorbar for the contour lines
#CB = plt.colorbar(CS, shrink=0.8, extend='both')

plt.show()


