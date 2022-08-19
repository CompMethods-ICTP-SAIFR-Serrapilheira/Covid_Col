# ------------------------------------------------------#
# Scientific computing
# ICTP/Serrapilheira 2022
# Final project: COVID19 Colombia
# First version 2022-08-15
# ------------------------------------------------------#

import numpy as np
import math

# Make the transition between states in each time step
def pomp_transition(var, rate, dt = 1, num_ensembles = 200, kb = None):
    if kb is None:
        kb = np.maximum(1.0 - np.exp(-rate*dt), 0)
    num_ind   = np.random.binomial(list(var), kb )
    return np.squeeze(num_ind)
  
# Initialize the model for unvaccinated scenario
def init_modelUV(pop, num_ensembles = 300, num_variables = 8, num_age_groups = 16):
    x_init = np.zeros((num_variables, num_ensembles))
    for idx_ens in range(num_ensembles):
        S0   = pop
        E0   = 3
        Ir0  = 1
        Iu0  = 1
        Id0  = 0
        R0   = 0
        SV0   = 0
        EV0   = 0
        IVr0  = 0
        IVu0  = 0
        S0   = S0-E0-Ir0-Iu0-SV0-EV0-IVr0-IVu0-Id0-R0
        
        C0   = Ir0
        D0   = 0
        
        IVd0  = 0
        RV0   = 0
        CV0   = IVr0
        DV0   = 0

        x_init[:,idx_ens] = [S0, E0, Ir0, Iu0, Id0, R0, C0, D0, SV0, EV0, IVr0, IVu0, IVd0, RV0, CV0, DV0]

    return x_init
# Initialize the model for vaccinated scenario
def init_modelV(pop, num_ensembles = 300, num_variables = 8, num_age_groups = 16):

    x_init = np.zeros((num_variables, num_ensembles))
    for idx_ens in range(num_ensembles):
        S0   = pop
        E0   = 1250932
        Ir0  = 179620
        Iu0  = 810287
        Id0  = 20579
        R0   = 26626090
        SV0   = 0
        EV0   = 0
        IVr0  = 0
        IVu0  = 0
        S0   = S0-E0-Ir0-Iu0-SV0-EV0-IVr0-IVu0-Id0-R0
        
        C0   = Ir0
        D0   = Id0/12
        
        IVd0  = 0
        RV0   = 0
        CV0   = IVr0
        DV0   = 0

        x_init[:,idx_ens] = [S0, E0, Ir0, Iu0, Id0, R0, C0, D0, SV0, EV0, IVr0, IVu0, IVd0, RV0, CV0, DV0]

    return x_init

# COVID19 Model.
def modelV(x, beta, ifr, iVfr, Vr, alpha, N, num_ensembles = 200):

    
    kappa   = 1/6.4 # Incubation Period [days]
    gamma   = 1/4   # Recovery Period [days]
    gamma_d = 1/12  # Death Period [days]
    sigma   = 0.5   # Relative Unreported tranmissibility [Adimensional]
    VEI = (0.762)   # One doses vaccine efficacy in reducing infectivity [Adimensional]
    
    delta = 1/182

    S   = x[0,:]
    E   = x[1,:] # List with Exposed individuals in all age groups
    Ir  = x[2,:] # List with reported Infected Individuals in all age groups
    Iu  = x[3,:] # List with Un-reported Infected individuals in all age groups
    Id  = x[4,:] # List with Infected individuals who eventually will die (We)
    R   = x[5,:] # List with Recovered and transient inmmune individuals
    C   = x[6,:] # List with incident cases in all age groups
    D   = x[7,:] # List with incident Deaths in all age groups

    SV  = x[8,:]
    EV  = x[9,:]  # List with Exposed individuals with one vaccine dose in all age groups
    IVr = x[10,:] # List with reported Infected Individuals with one vaccine dose in all age groups
    IVu = x[11,:] # List with Un-reported Infected individuals with one vaccine dose in all age groups
    IVd = x[12,:] # List with Infected individuals with one vaccine dose who eventually will die (We)
    RV  = x[13,:] # List with Recovered and transient inmmune individuals with one vaccine dose
    CV  = x[14,:] # List with incident cases with one vaccine dose in all age groups
    DV  = x[15,:] # List with incident Deaths with one vaccine dose in all age groups



    # Compute force of infection in each age group FOI_a | Just proportional to the number of infected reported individuals and the number of infected under-reported individuals
    foi = beta * (((Ir + sigma*Iu)+((1-VEI)*(IVr + sigma*IVu)))/ N)   
    
    ###### TRANSIIONS ######
    s2e      = pomp_transition(S, foi, num_ensembles = num_ensembles)                 # susceptible to exposed
    s2sV     = pomp_transition(Vr, 0, kb = np.nan_to_num(S/(S+E+R)), num_ensembles = num_ensembles)
    e2eV     = pomp_transition(Vr, 0, kb = np.nan_to_num(E/(S+E+R)), num_ensembles = num_ensembles)
    e2iu     = pomp_transition(E, (1-alpha)*kappa, num_ensembles = num_ensembles)     # exposed to infected underreported
    e2ir     = pomp_transition(E, alpha*(1-ifr)*kappa, num_ensembles = num_ensembles) # exposed to infected reported who are not going to die
    e2id     = pomp_transition(E, alpha*ifr*kappa, num_ensembles = num_ensembles)           # exposed to infected reported who are going to die
    iu2r     = pomp_transition(Iu, gamma, num_ensembles = num_ensembles)              # infected under-reported to recovered
    ir2r     = pomp_transition(Ir, gamma, num_ensembles = num_ensembles)              # infected reported (who are not going to die) to recovered
    r2rV     = pomp_transition(Vr, 0, kb = np.nan_to_num(R/(S+E+R)), num_ensembles = num_ensembles)
    id2death = pomp_transition(Id, gamma_d, num_ensembles = num_ensembles)            # infected reported (who are going to die) to Death
    r2s      = pomp_transition(R, delta, num_ensembles = num_ensembles)
    
    sV2eV      = pomp_transition(SV, foi, num_ensembles = num_ensembles)                  # susceptible to exposed
    eV2iVu     = pomp_transition(EV, (1-(alpha))*kappa, num_ensembles = num_ensembles)      # exposed to infected underreported
    eV2iVr     = pomp_transition(EV, alpha*(1-iVfr)*kappa, num_ensembles = num_ensembles) # exposed to infected reported who are not going to die
    eV2iVd     = pomp_transition(EV, alpha*iVfr*kappa, num_ensembles = num_ensembles)           # exposed to infected reported who are going to die
    iVu2rV     = pomp_transition(IVu, gamma, num_ensembles = num_ensembles)               # infected under-reported to recovered
    iVr2rV     = pomp_transition(IVr, gamma, num_ensembles = num_ensembles)               # infected reported (who are not going to die) to recovered
    iVd2deathV = pomp_transition(IVd, gamma_d, num_ensembles = num_ensembles)             # infected reported (who are going to die) to Death
    rV2sV      = pomp_transition(RV, delta, num_ensembles = num_ensembles)
    
    S  = S    - s2e  - s2sV + r2s                # Susceptible
    E  = E    + s2e  - e2ir - e2iu - e2id - e2eV # Exposed
    Ir = Ir   + e2ir - ir2r                      # Infected reported
    Iu = Iu   + e2iu - iu2r                      # Infected un-reported
    Id = Id   + e2id - id2death                  # Infected who are going to die
    R  = R    + ir2r + iu2r - r2rV - r2s         # Recovered
    C  = e2ir + e2id                             # Incident Cases
    D  = id2death                                # Incident Deaths

    SV  = SV  - sV2eV  + s2sV  + rV2sV                   # Susceptible
    EV  = EV  + sV2eV  - eV2iVr - eV2iVu - eV2iVd + e2eV # Exposed
    IVr = IVr + eV2iVr - iVr2rV                          # Infected reported
    IVu = IVu + eV2iVu - iVu2rV                          # Infected un-reported
    IVd = IVd + eV2iVd - iVd2deathV                      # Infected who are going to die
    RV  = RV  + iVr2rV + iVu2rV + r2rV - rV2sV           # Recovered
    CV  = eV2iVr + eV2iVd                                # Incident Cases
    DV  = iVd2deathV                                     # Incident Deaths
  

    return np.array([S, E, Ir, Iu, Id, R, C, D, SV, EV, IVr, IVu, IVd, RV, CV, DV])
