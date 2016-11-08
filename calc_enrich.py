import numpy as np
import pandas as pd

# Constants
D_rho = 2.2e-5 # kg/ms
R = 8.314 # J/K*mol
dM = 0.003 #kg/mol
M = 0.352 # kg/mol of UF6

# Centrifuge assumptions
x = 1000 # pressure ratio (Glaser)
T = 320.0   # K
cut = 0.50
k = 2.0  # L/F ratio

# Centrifuge parameters
v_a = 485.0 # m/s
Z = 1.0   # m
d = 0.15  # m 
F_flow = 15e-6 # kg/s (paper is in mg/s)

a = d/2.0 # outer radius
r_2 = 0.99*a  # fraction of a

def calc_del_U():
    
    # Centrifuge assumptions
    #    Z = 2 # m
    #    r2 = 0.96  # Range 0.96 - 0.99 (Glaser)
    #    T = 273 + 70 # Celsius to kelvin 
    #    r_a = 0.05 #m
    #    omega = 63000/60 #rpm -> rot/sec
    #v_a = omega*r_a

    L_flow = k*F_flow

    # Intermediate calculations
    r_12 = np.sqrt(1.0 - (2.0*R*T*(np.log(x))/M/(v_a**2))) # fraction
    r_1 = r_2*r_12  # fraction
    
    L_F = L_flow/F_flow  #range 2-4
    Z_p = Z*(1.0 - cut)*(1.0 + L_F)/(1.0 - cut + L_F)

    print "L_F= ", L_F
    print "Z_p=  ", Z_p
    
    C1 = (2.0*np.pi*D_rho/(np.log(r_2/r_1)))
    A_p = C1 *(1.0/F_flow) * (cut/((1.0 + L_F)*(1.0 - cut + L_F)))
    A_w = C1 * (1.0/F_flow) * ((1.0 - cut)/(L_F*(1.0 - cut + L_F)))

    print "C1= ", C1
    print "A_p=  ", A_p
    print "A_w=  ", A_w

    
    C_flow = 0.5*F_flow*cut*(1.0 - cut)
    C_therm = (dM * (v_a**2))/(2.0 * R * T)
    print "C_flow=  ", C_flow
    print "C_therm=  ", C_therm

    C_scale = ((r_2/a)**4)*((1-(r_12**2))**2)
    bracket1 = (1 + L_F)/cut
    exp1 = np.exp(-1.0*A_p*Z_p)
    bracket2 = L_F/(1 - cut)
    exp2 = np.exp(-1.0*A_w*(Z - Z_p))

    print "C_scale=  ", C_scale
    print "bracket1= ", bracket1
    print "bracket2= ", bracket2
    print "exp1= ", np.exp(exp1)
    print "exp2= ", np.exp(exp2)

    
    del_U = 0.5*F_flow*cut*(1.0 - cut)*(C_therm**2)*C_scale*(
        (bracket1*(1 - exp1)) + (bracket2*(1 - exp2)))**2 # kg/s
        
    per_sec2yr = 60*60*24*365.25 # s/m * m/hr * hr/d * d/y

    dirac = 0.5*np.pi*Z*D_rho*C_therm*per_sec2yr  # kg/s
    del_U = del_U * per_sec2yr

    alpha = np.exp(np.sqrt(2)*C_therm*Z/d)
    
    return alpha, del_U, dirac


def calc_V(N_in):
    V_out = (2.0*N_in - 1.0)*np.log(N_in/(1.0 - N_in))
    return V_out
    
# **** NOT WORKING ***
def calc_alpha():
    # define Feed, del_U from above
    # assume one of Np, Nf, Nw

    # Define NF
    alpha = 1.0 + ((2.0*cut -1)/(Nf - 0.5))
    Np = 1.0/(1+ ((1-Nf)/(alpha*Nf)))
    Nw = (Nf - cut*Np)/(1.0 - cut)

    Vf = calc_V(Nf)
    Vp = calc_V(Np)
    Vw = calc_V(Nw)

    # Create a table of Np, Vp
    n_steps = 10
    V = []
    for x in range(0, n_steps):
        real_x = x/n_steps
        value = (2.0*x - 1)*np.log(x/(1.0 - x))
        V.append(value)

    
#    del_U = F*((cut*Vp) + ((1 - cut)* Vw) - Vf)


# *** UNTESTED ****
def n_stages(alpha=1.04, Nf=0.007, Np=0.20, Nw=0.002):
    enrich_inner = (Np/(1.0 - Np))*((1.0 - Nf)/Nf)
    enrich_stages = (1.0/(alpha-1.0))*np.log(enrich_inner)
    strip_inner =  (Nf/(1.0 - Nf))*((1.0 - Nw)/Nw)
    strip_stages = (1.0/(alpha-1.0))*np.log(strip_inner)

    return enrich_stages, strip_stages

# *** UNTESTED ****
def cf_per_stage(Npc, Nwc, Nfs, Pc, alpha):
    # Npc, Nwc, Nfc = enrichment of cascade product/waste/feed
    # Nfs, Nws, Nps = enrichment of stage product/waste/feed
    # Pc = cascade product flow
    
    epsilon = 1.0 - alpha
    # L_stage = counter-current flow of stage
    L_stage_enrich = 2*Pc*(Npc - Nfs)/(epsilon*Nfs*(1-Nfs)) # Check book?
    L_stage_strip = 2*Wc*(Nfs - Nwc)/(epsilon*Nfs*(1-Nfs)) # Check book?

    Vpc = calc_V(Npc)
    Vwc = calc_V(Nwc)
    
    ## WHAT IS EQN FOR TOTAL NUMBER OF STAGES???
    n_cf = 2*(Pc*Vpc + Wc*Vwc)/(L_stage_tot*(epsilon**2)) # CHECK BOOK

    return n_cf, L_stage_tot