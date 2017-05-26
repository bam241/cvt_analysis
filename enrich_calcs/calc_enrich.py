import numpy as np
import pandas as pd

# Constants
D_rho = 2.2e-5 # kg/ms
R = 8.314 # J/K*mol
dM = 0.003 #kg/mol
M = 0.352 # kg/mol of UF6
M_atm = 0.238 # atomic mass of natural uranium

# Centrifuge assumptions
x = 1000 # pressure ratio (Glaser)
k = 2.0  # L/F ratio

# Centrifuge parameters
#v_a = 485.0 # m/s
#Z = 1.0   # m
#d = 0.15  # m 
#F_m = 15e-6 # kg/s (paper is in mg/s)
#r_12 = 0.746

def calc_del_U(v_a, Z, d, F_m, T, cut, eff=1.0, verbose=False):
    a = d/2.0 # outer radius
    r_2 = 0.975*a  # fraction of a
    

    # Intermediate calculations
    r_12 = np.sqrt(1.0 - (2.0*R*T*(np.log(x))/M/(v_a**2))) # fraction
    r_1 = r_2*r_12  # fraction

    # Glaser eqn 12
    L_F = k  #range 2-4
    Z_p = Z*(1.0 - cut)*(1.0 + L_F)/(1.0 - cut + L_F)

    if (verbose == True):
        print "L_F= ", L_F
        print "Z_p=  ", Z_p
        print "r_1", r_1
        print "r_12", r_12

    # Glaser eqn 3
    # To convert from gas to atom fraction, multiple by M_atm/M 
    C1 = (2.0*np.pi*(D_rho*M_atm/M)/(np.log(r_2/r_1)))
#    C1 = (2.0*np.pi*(D_rho)/(np.log(r_2/r_1)))
    A_p = C1 *(1.0/F_m) * (cut/((1.0 + L_F)*(1.0 - cut + L_F)))
    A_w = C1 * (1.0/F_m) * ((1.0 - cut)/(L_F*(1.0 - cut + L_F)))

    C_flow = 0.5*F_m*cut*(1.0 - cut)
    C_therm = calc_C_therm(v_a, T)

    C_scale = ((r_2/a)**4)*((1-(r_12**2))**2)
    bracket1 = (1 + L_F)/cut
    exp1 = np.exp(-1.0*A_p*Z_p)
    bracket2 = L_F/(1 - cut)
    exp2 = np.exp(-1.0*A_w*(Z - Z_p))

    # Glaser eqn 10
    # Efficiency applied to optimal del_U in Ratz p73 (pdf p21)
    major_term = 0.5*cut*(1.0 - cut)*(C_therm**2)*C_scale*(
                (bracket1*(1 - exp1)) + (bracket2*(1 - exp2)))**2 # kg/s    
    del_U = F_m*major_term*eff #kg/s
    
    per_sec2yr = 60*60*24*365.25 # s/m * m/hr * hr/d * d/y

    # Glaser eqn 6
    dirac = 0.5*np.pi*Z*(D_rho*M_atm/M)*(C_therm**2)*per_sec2yr  # kg/s
    del_U_yr = del_U * per_sec2yr

    # Avery p.18
    alpha = alpha_by_swu(del_U, F_m, cut)
    
    return alpha, del_U, del_U_yr, dirac  #kg/sec

# for a machine
def calc_C_therm(v_a, T):
    C_therm = (dM * (v_a**2))/(2.0 * R * T)
    return C_therm

def calc_V(N_in):
    V_out = (2.0*N_in - 1.0)*np.log(N_in/(1.0 - N_in))
    return V_out

# for a machine
def alpha_by_swu(del_U, F_m, cut):
    # Avery p.18
    # del_U in moles/sec
    del_U_moles = del_U/M
    alpha = 1 + np.sqrt((2*del_U_moles*(1-cut)/(cut*F_m)))
    return alpha

# for a machine
def alpha_by_enrich(Nf, Np):
    num = Np/(1 - Np)
    denom = Nf/(1-Nf)
    alpha_enr = num/denom
    return alpha_enr

# for a machine
# ** IS C_therm supposed to be squared here?
def alpha_max_theory(v_a, Z, d, T):
    # Avery p36
    # Max theoretical separation for zero withdrawal
    C_therm = calc_C_therm(v_a, T)
    alpha_th = np.exp(np.sqrt(2)*C_therm*Z/d)

# for a machine
# Avery p. 57
def N_product_by_alpha(alpha, Nfm):
#    ratio = (1.0 - Nfm)/(alpha*Nfm)
#    Npm = 1.0/(ratio + 1.0)
    ratio = alpha*Nfm/(1.0 - Nfm)
    Npm = ratio/(1+ratio)
    return Npm

# for a machine
# Avery p.59
# IN THE LIMIT WHERE ALPHA ->1
def N_waste_by_alpha(alpha, Nfm):
    A = (Nfm/(1-Nfm))/alpha
    Nwm = A/(1+A)
    return Nwm

## This equation can only be used in the limit where the separation factor
## (alpha) is very close to one, which is not true for modern gas centrifuges
## DO NOT USE THIS EQUATION!!!
# Avery p.59
def stages_per_cascade(alpha, Nfc, Npc, Nwc):
    epsilon = alpha - 1.0
    enrich_inner = (Npc/(1.0 - Npc))*((1.0 - Nfc)/Nfc)
    strip_inner =  (Nfc/(1.0 - Nfc))*((1.0 - Nwc)/Nwc)

    enrich_stages = (1.0/epsilon)*np.log(enrich_inner)
    strip_stages = (1.0/epsilon)*np.log(strip_inner)

    return enrich_stages, strip_stages

# derived from Avery ??
def Npc_from_Nstages(alpha, Nfc, enrich_stages):
    epsilon = alpha - 1.0
    A = (Nfc/(1 - Nfc))*np.exp(enrich_stages*epsilon)
    Npc = A/(1 + A)
    return Npc
    
def Nwc_from_Nstages(alpha, Nfc, strip_stages):
    epsilon = alpha - 1.0
    B = ((1 - Nfc)/Nfc)*np.exp(strip_stages*epsilon)
    Nwc = 1/(1 + B)
    return Nwc

def machines_per_enr_stage(alpha, del_U, Fs):
    # flows do not have required units so long as they are consistent
    # Nfs, Nws, Nps = enrichment of stage product/waste/feed
    
    epsilon = alpha - 1.0

    # Feed flow of a single machine (in Avery denoted with L)
    # Avery p. 62
    F_machine = 2.0*del_U/(epsilon**2)
    n_enrich = Fs/F_machine
    return n_enrich

# assuming feed into machines is already at its maximum,
# total throughput is limited by number of available machines
# (in a system where there are insufficient total machines)
def allowed_feed_per_stage(alpha, del_U, n_mach):
    max_feed = (n_mach*2*del_U)/((alpha - 1)**2)
    return max_feed

def product_per_enr_stage(alpha, Nfs, Nps, Fs):
    epsilon = alpha - 1.0

    # F_stage = incoming flow (in Avery denoted with L_r)
    # Avery p. 60
    Ps_enrich = Fs*epsilon*Nfs*(1 - Nfs)/(2*(Nps - Nfs))

    return Ps_enrich

## 14-Feb-2017
## although in Avery this eqn depends on waste_per_strip_stage, which is
## wrong, I have implemented it based only on ratio of total stage flow
## to machine flow, so it is correct
def machines_per_strip_stage(alpha, del_U, Fs):
    # flows do not have required units so long as they are consistent

    epsilon = alpha - 1.0
    
    # Feed flow of a single machine (in Avery denoted with L)
    # Avery p. 62
    F_machine = 2.0*del_U/(epsilon**2)
    n_strip = Fs/F_machine

    return n_strip

## 14-Feb-2017
## THIS EQN PRODUCES THE WRONG RESULT FOR SOME REASON.
## DONT KNOW WHAT THE PROBLEM IS THOUGH
def waste_per_strip_stage(alpha, Nfs, Nws, Fs):
    epsilon = alpha - 1.0

    # F_stage = incoming flow (in Avery denoted with L_r)
    # Avery p. 60
    W_strip = Fs*epsilon*Nfs*(1 - Nfs)/(2*(Nfs - Nws))

    return  W_strip

def delta_U_cascade(Npc, Nwc, Fc, Pc):
    Vpc = calc_V(Npc)
    Vwc = calc_V(Nwc)
    Wc = Fc - Pc
    delta_U_cascade = Pc*Vpc + Wc*Vwc

    return delta_U_cascade

def machines_per_cascade(del_U_machine, Npc, Nwc, Fc, Pc):
    # Avery p 62
    U_cascade = delta_U_cascade(Npc, Nwc, Fc, Pc)
    n_cf = U_cascade/del_U_machine

    return n_cf

def del_U_by_cascade_config(Npc, Nwc, Pc, Wc, n_cf):
    U_cascade = delta_U_cascade(Npc, Nwc, Pc, Wc)
    del_U_machine = U_cascade/n_cf

    return del_U_machine

# Determines the total feed flow rates required at each stage
# of the cascade for steady-state flow
def calc_feed_flows(n_stages_enrich, n_stages_strip, cascade_feed, cut):
    n_stages = n_stages_strip + n_stages_enrich

    eqn_answers = np.zeros(n_stages)
    for i in range(-1*n_stages_strip, n_stages_enrich):
        eqn = np.zeros(n_stages)
        position = n_stages_strip + i
        eqn[position] = -1
        if (position != 0):
            eqn[position - 1] = cut
        if (position != n_stages - 1):
            eqn[position + 1] = (1-cut)
        if (position == 0):
            eqn_array = eqn
        else:
            eqn_array = np.vstack((eqn_array,eqn))
        if (i == 0):
            eqn_answers[position] = -1*cascade_feed

    return np.linalg.solve(eqn_array, eqn_answers)

def find_N_stages(alpha, feed_assay, product_assay, waste_assay):
    ideal_enrich_stage = 0
    ideal_strip_stage = 0
    Nfs = feed_assay
    Nps = feed_assay

    while (Nps < product_assay):
        Nps = N_product_by_alpha(alpha, Nfs)
        if (ideal_enrich_stage == 0):
            Nws = N_waste_by_alpha(alpha, Nfs)
        ideal_enrich_stage +=1
        Nfs = Nps

    Nfs = Nws
    while (Nws > waste_assay):
        Nws = N_waste_by_alpha(alpha, Nfs)
        ideal_strip_stage += 1
        Nfs = Nws

    return ideal_enrich_stage, ideal_strip_stage

def design_cascade(cut, alpha, del_U, Nfc, feed_flows,
                   ideal_enrich_stage, ideal_strip_stage,
                   assay_len=4, qty_len=6, verbose=False, pretty=False):

    print_len = qty_len
    fix = 1 # all flows are same units as input feed_flows
    # For verbose only, make print statements in useful units
    if (pretty == True):
        fix = 30.4*24*60*60  # convert from kg/sec to kg/mon
        print_len = 2
        
    if (verbose == True):
        print "Stage   #Mach\t Feed    Product  Waste\t F_assay \tP_assay W_assay"

    n_centrifuges = 0
    Nfs = Nfc
    all_stages = []
    Nw_1 = 0
    for stage_idx in range(ideal_enrich_stage):
        curr_stage = stage_idx + ideal_strip_stage
        Fs = feed_flows[curr_stage]
        # Truncate to integer and then add 1 to ensure enough capacity
        # to preserve steady-state flow rates input
        n_mach_enr = int(machines_per_enr_stage(alpha, del_U, Fs)) + 1
        Nps = N_product_by_alpha(alpha, Nfs)
        Ps = Fs*cut
        Ws = Fs - Ps
        Nws = N_waste_by_alpha(alpha, Nfs)
        all_stages.append([stage_idx, n_mach_enr, Fs, 
                           Ps, Ws,
                           Nfs, Nps,
                           Nws])
        n_centrifuges += n_mach_enr
        if (stage_idx == 0):
            Nw_1 = Nws

        if (verbose == True):
             print stage_idx, "\t", n_mach_enr,"\t", round(Fs*fix,print_len), "  ",round(Ps*fix, print_len), "  ",round(Ws*fix, print_len),"  ", round(Nfs, assay_len), "\t",round(Nps, assay_len),"\t", round(Nws, assay_len)

        Nfs = Nps

    Nfs = Nw_1
    for stage_idx in range(ideal_strip_stage-1,-1,-1):
        curr_stage = stage_idx - ideal_strip_stage
        Fs = feed_flows[stage_idx]
        # Truncate to integer and then add 1 to ensure enough capacity
        # to preserve steady-state flow rates input
        n_mach_strip = int (machines_per_strip_stage(alpha, del_U, Fs)) + 1
        Nps = N_product_by_alpha(alpha, Nfs)
        Nws = N_waste_by_alpha(alpha, Nfs)
        Ps = Fs * cut
        Ws = Fs - Ps
        all_stages.insert(0,[(curr_stage), n_mach_strip, Fs,
                             Ps, Ws,
                             Nfs, 
                             Nps, Nws])
        n_centrifuges += n_mach_strip

        if (verbose == True):
            print (curr_stage), "\t",n_mach_strip ,"\t", round(Fs*fix,print_len), "  ",round(Ps*fix, print_len), "  ",round(Ws*fix, print_len),"  ", round(Nfs, assay_len), "\t",round(Nps, assay_len),"\t", round(Nws, assay_len)
        Nfs = Nws

    return all_stages, n_centrifuges