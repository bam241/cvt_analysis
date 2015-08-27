#!/usr/bin/env python


import cymetric as cym
import pandas as pd
import numpy as np
import math
from cymetric import root_metric


#frame = cym.eval('Transactions', db, conds=[('Commodity', '==', 'spent_uox', 'RecevierId' > 14)])
#commods = frame[:]['Commodity'].unique()
#filtered_frame = cym.eval('Materials', db, conds=[('NucId', '==', 922350000)])


def facility_input(dbname):
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
    rows = 20
    cols = rows
    src = [0,1]   # [row,col]
#    n_times = 18
#    curr_emiss = np.ndarray(n_times)
#    curr_emiss.fill(0)
#    curr_emiss[0] = 10000
#    curr_emiss[5] = 5000
#    curr_emiss[10] = 50

    mn=1e-1

    
    pos_matrix = np.ndarray((rows,cols))
    pos_matrix.fill(0)
    pos_matrix[src[0]][src[1]] = 1
    
    inventory_frame = facility_input(db)
    n_times = inventory_frame['TimeCreated'].size
    curr_emiss = inventory_frame['17']  # 18 = divertor

    tot_emiss = np.ndarray((n_times,rows,cols))
    tot_emiss.fill(mn)
    for r in range(rows):
        for c in range(cols):
            if (r == src[0] and c == src[1]):
                dist = 0
            else:
                dist = math.sqrt((r - src[0])**2 + (c - src[1])**2)
            pos_matrix[r][c] = dist
    fall_off_matrix = 1/((1+pos_matrix)**2)
    time_delay_matrix = pos_matrix/velocity
    print("fall_off value is: ")
    print(fall_off_matrix[0][1])

    print("time_delay_matrix is: ")
    print(time_delay_matrix[0][1])
    
    for t in range(n_times):
        em = curr_emiss[t]
        if (em > mn):
            eff_emiss = em*fall_off_matrix
            for r in range(rows):
                for c in range(cols):
                    eff_time_delay = time_delay_matrix[r][c]
                    if (t + eff_time_delay) < n_times:  # change for real program
                        tot_emiss[t+eff_time_delay][r][c] += eff_emiss[r][c]
    return tot_emiss


#def emissions_movie(dbfile):
dir = '/Users/mbmcgarry/git/data_analysis/data/testing/random_sep/'
dbfile = dir+'norm_dist.sqlite'

import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from matplotlib import colors


matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'


ps=1

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

fig = plt.figure()
plt.xlabel('x location')
plt.ylabel('y location')
min_em = np.log(np.amin(emissions, axis=2).min())
#max_em = np.amax(emissions, axis=2).max()
max_em = np.log(np.amax(emissions, axis=2).max())
n_lev = (max_em - min_em)/50
levels = np.arange(min_em, max_em, n_lev)
#levels = np.logspace(min_em, max_em+1, 20)
print(min_em, max_em)

def make_frame(i):
    Z = emissions[i][:][:]
    plt.clf()
#    cont = plt.contourf(np.log(Z), interpolation='bilinear', origin='lower', levels=levels,
    cont = plt.contourf(np.log(Z), interpolation='bilinear', origin='lower',
                        levels=levels,
#                        norm=matplotlib.colors.LogNorm(),
                        extent=(xmin-edge,xmax+edge,ymin-edge,ymax+edge))
    plt.title("Kr85 emissions, t="+ str(i))
    CB = plt.colorbar(cont, shrink=0.8, extend='both')
#                format='%.2f',ticks=[min_em, max_em])
#                      format='%.2f',ticks=[np.log(min_em), np.log(max_em)])
#    CB.ax.set_yticklabels([math.exp(min_em),math.exp(max_em)])
    CB.ax.set_ylabel('ln(I)')
    return cont

fps=2.0
anim=1
if anim == 1:
    ax = fig.add_subplot(111)
    anim = animation.FuncAnimation(fig, make_frame, frames=n_times,
                                   repeat_delay=2000, interval=(1e3)/fps,
                                   blit=False) 
else:
    cont = plt.imshow(emissions[5][:][:], interpolation='bilinear', origin='lower',
                      extent=(xmin-edge,xmax+edge,ymin-edge,ymax+edge))
    CB = plt.colorbar(cont, shrink=0.8, extend='both', vmin=min_em, vmax=max_em)


if ps == 1:
    #Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(dir+'one_source.mp4', writer=writer)
else:
    plt.show()

#    return(anim)



