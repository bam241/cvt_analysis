import numpy as np
import pandas as pd

# Constants
D_rho = 2.2e-5 # kg/ms
R = 8.314 # J/K*mol
dM = 0.003 #kg/mol
M = 0.352 # kg/mol of UF6

# Centrifuge assumptions
x = 1000 # pressure ratio (Glaser)
k = 2.0  # L/F ratio

# Centrifuge parameters
#v_a = 485.0 # m/s
#Z = 1.0   # m
#d = 0.15  # m 
#F_m = 15e-6 # kg/s (paper is in mg/s)


def calc_del_U(v_a, Z, d, F_m, T, cut, eff=1.0):
    a = d/2.0 # outer radius
    r_2 = 0.99*a  # fraction of a
    

    # Intermediate calculations
    r_12 = np.sqrt(1.0 - (2.0*R*T*(np.log(x))/M/(v_a**2))) # fraction
    r_1 = r_2*r_12  # fraction

    # Glaser eqn 12
    L_F = k  #range 2-4
    Z_p = Z*(1.0 - cut)*(1.0 + L_F)/(1.0 - cut + L_F)

    print "L_F= ", L_F
    print "Z_p=  ", Z_p

    # Glaser eqn 3
    C1 = (2.0*np.pi*D_rho/(np.log(r_2/r_1)))
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
    major_term = 0.5*cut*(1.0 - cut)*(C_therm**2)*C_scale*(
                (bracket1*(1 - exp1)) + (bracket2*(1 - exp2)))**2 # kg/s    
    del_U = F_m*major_term*eff #kg/s
    
    per_sec2yr = 60*60*24*365.25 # s/m * m/hr * hr/d * d/y

    # Glaser eqn 6
    dirac = 0.5*np.pi*Z*D_rho*C_therm*per_sec2yr  # kg/s
    del_U_yr = del_U * per_sec2yr

    # Avery p.18
    alpha = alpha_by_swu(del_U, F_m, cut)
    
    return alpha, del_U, del_U_yr  #kg/sec

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
def alpha_max_theory(v_a, Z, d, T):
    # Avery p36
    # Max theoretical separation for zero withdrawal
    C_therm = calc_C_therm(v_a, T)
    alpha_th = np.exp(np.sqrt(2)*C_therm*Z/d)

# for a machine
def N_product_by_alpha(alpha, Nfm):
    ratio = (1.0 - Nfm)/(alpha*Nfm)
    Npm = 1.0/(ratio + 1.0)
    return Npm

# for a machine
# Avery p.59
def N_waste_by_alpha(alpha, Nfm):
    A = (Nfm/(1-Nfm))/alpha
    Nwm = A/(1+A)
    return Nwm
    
def stages_per_cascade(alpha, Nfc, Npc, Nwc):
    epsilon = alpha - 1.0
    enrich_inner = (Npc/(1.0 - Npc))*((1.0 - Nfc)/Nfc)
    strip_inner =  (Nfc/(1.0 - Nfc))*((1.0 - Nwc)/Nwc)

    enrich_stages = (1.0/epsilon)*np.log(enrich_inner)
    strip_stages = (1.0/epsilon)*np.log(strip_inner)

    return enrich_stages, strip_stages


def machines_per_enr_stage(alpha, del_U, Nfs, Nps, Fs):
    # flows do not have required units so long as they are consistent
    # Nfs, Nws, Nps = enrichment of stage product/waste/feed
    
    epsilon = alpha - 1.0

    # Feed flow of a single machine (in Avery denoted with L)
    # Avery p. 62
    F_machine = 2.0*del_U/(epsilon**2)
    n_enrich = Fs/F_machine

    # F_stage = incoming flow (in Avery denoted with L_r)
    # Avery p. 60
    Ps_enrich = Fs*epsilon*Nfs*(1 - Nfs)/(2*(Nps - Nfs))

    return n_enrich, Ps_enrich

def machines_per_strip_stage(alpha, del_U, Nfs, Nws, Fs):
    # flows do not have required units so long as they are consistent

    epsilon = alpha - 1.0
    
    # Feed flow of a single machine (in Avery denoted with L)
    # Avery p. 62
    F_machine = 2.0*del_U/(epsilon**2)
    n_strip = Fs/F_machine

    # F_stage = incoming flow (in Avery denoted with L_r)
    # Avery p. 60
    W_strip = Fs*epsilon*Nfs*(1 - Nfs)/(2*(Nfs - Nws))

    return  n_strip, W_strip

def delta_U_cascade(Npc, Nwc, Fc, Pc):
    Vpc = calc_V(Npc)
    Vwc = calc_V(Nwc)
    Wc = Fc - Pc
    delta_U_cascade = Pc*Vpc + Wc*Vwc

    return delta_U_cascade

def machines_per_cascade(del_U_machine, Npc, Nwc, Pc, Wc):
    # Avery p 62
    U_cascade = delta_U_cascade(Npc, Nwc, Pc, Wc)
    n_cf = U_cascade/del_U_machine

    return n_cf

def del_U_by_cascade_config(Npc, Nwc, Pc, Wc, n_cf):
    U_cascade = delta_U_cascade(Npc, Nwc, Pc, Wc)
    del_U_machine = U_cascade/n_cf

    return del_U_machine

