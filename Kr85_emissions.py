#!/usr/bin/env python


import cymetric as cym
import pandas as pd
import numpy as np
import math
from cymetric import root_metric


def facility_input(dbname):
    db = cym.dbopen(dbname)

    frame = cym.eval('TransactionQuantity', db)

    ids = frame[:]['ReceiverId'].unique()
    ids.sort()

    it = 0
    for cur_id in ids:
        cur_frame = cym.eval('TransactionQuantity', db,
                             conds=[('ReceiverId', '==', cur_id)])
        raw_time = cur_frame[:]['TimeCreated']
        raw_data = cur_frame[:]['Quantity']
        n_entries = raw_time.size
        n_times = raw_time[n_entries - 1]+1
        # if not every time step is represented in the dataframe
        if n_entries != n_times:
            index = np.arange(n_times)
            time_col = ['TimeCreated']
            data_col = ['Quantity']                
            corr_time = pd.DataFrame(columns=time_col, index=index)
            corr_data = pd.DataFrame(columns=data_col, index=index)
            i_last = 0
            for elem, val in raw_time.iteritems():
                # if the current index is not equal to the current timestep
                if val != i_last:
                    n_missing = val - i_last
                    for t_miss in xrange(1 - n_missing, 0):
                        corr_time.ix[val + t_miss] = val + t_miss
                        corr_data.ix[val + t_miss] = 0
                    i_last+= n_missing
                    corr_time.ix[val] = val
                    corr_data.ix[val] = raw_data[elem]
                else:
                    corr_time.ix[val] = raw_time[elem]
                    corr_data.ix[val] = raw_data[elem]
        else:
            corr_data = raw_data
            corr_time = raw_time

        if it == 0:
            big_frame = corr_time

        big_frame = pd.concat([big_frame, corr_data], axis=1)
        it += 1

    str_fac = map(str,ids)
    big_frame.columns = ['TimeCreated']+str_fac
    big_frame.fillna(0)  # in case one time series ends earlier than the others
    return big_frame


def calc_emissions(db, fac_info, mn):
    # fac_info is a dictionary where each key is the facility id
    # and it's value is the location in row, col position 
    #released = {
    #		"17" : [0,1]   # for separations stream output
    #    }
    
    velocity = 0.6 # m/s
    rows = 20
    cols = rows

    inventory_frame = facility_input(db)
    n_times = inventory_frame['TimeCreated'].size

    tot_emiss = np.ndarray((n_times,rows,cols))
    tot_emiss.fill(mn)

    for facility, pos in fac_info.items():
        # Calculate the attenuation coefficient at each grid point based on distance
        # from source and effective time delay
        pos_matrix = np.ndarray((rows,cols))
        pos_matrix.fill(0)
        pos_matrix[pos[0]][pos[1]] = 1

        for r in range(rows):
            for c in range(cols):
                if (r == pos[0] and c == pos[1]):
                    dist = 0
                else:
                    dist = math.sqrt((r - pos[0])**2 + (c - pos[1])**2)
                pos_matrix[r][c] = dist
        fall_off_matrix = 1/((1+pos_matrix)**2)
        time_delay_matrix = pos_matrix/velocity

        # Add calculate the total signal at each grid point for each time for
        # each contributing emission source (facility)
        curr_emiss = inventory_frame[str(facility)]
        for t in range(n_times):
            em = curr_emiss[t]
            if (em > mn):
                eff_emiss = em*fall_off_matrix
                for r in range(rows):
                    for c in range(cols):
                        eff_time_delay = time_delay_matrix[r][c]
                        if (t + eff_time_delay) < n_times: 
                            tot_emiss[t+eff_time_delay][r][c] += eff_emiss[r][c]
    return tot_emiss

def emissions_movie(db, ps, mn):
    import matplotlib
    import numpy as np
    import matplotlib.cm as cm
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import math
    import os
    from matplotlib import colors

    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'

    fac_info = {
        13 : [1,10],   # RG_Sep1
        14 : [18,10],  # RG_Sep2
        15 : [9,10]    # WG_Sep
        }

    emissions = calc_emissions(db, fac_info, mn)
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

    def make_frame(i):
        Z = emissions[i][:][:]
        plt.clf()
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

#    fps=100.0   # rate to approx match mp4 playback
    fps= 5
    anim=1
    if anim == 1:
        ax = fig.add_subplot(111)
        anim = animation.FuncAnimation(fig, make_frame, frames=n_times,
                                       repeat_delay=2000, interval=(1e3)/fps,
                                       blit=False) 
    else:
        cont = plt.imshow(emissions[5][:][:], interpolation='bilinear',
                          origin='lower',
                          extent=(xmin-edge,xmax+edge,ymin-edge,ymax+edge))
        CB = plt.colorbar(cont, shrink=0.8, extend='both', vmin=min_em, vmax=max_em)


    if ps != 0:
        #Set up formatting for the movie files
        data_path, data_file = os.path.split(db)
        if not data_path:
            outfile = ps
        else:
            outfile = data_path + '/' + ps
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(outfile, writer=writer)
    else:
        plt.show()
        
    return(anim)



