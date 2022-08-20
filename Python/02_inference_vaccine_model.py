# ------------------------------------------------------#
# Scientific computing
# ICTP/Serrapilheira 2022
# Final project: COVID19 Colombia
# First version 2022-08-15
# ------------------------------------------------------#

# Import libraries and functions
from matplotlib.dates import date2num, num2date
from matplotlib.colors import ListedColormap
from matplotlib import dates as mdates
from matplotlib.patches import Patch
from matplotlib import pyplot as plt
from matplotlib import ticker
from scipy.stats import truncnorm
from tqdm import tqdm
import time

import pandas as pd
import numpy as np
import datetime
import math
import sys
import os

from functions.adjust_cases import prepare_cases
from scipy.interpolate import UnivariateSpline
from functions.model_agg import modelV, init_modelUV
from functions.utils_inference import eakf_step, checkbound_params, checkbound_state_vars, sample_params_uniform, sample_params_normal, inflate_ensembles

# define population
pop        = 40000000.0

# read cases by age without filtering
cases_df = pd.read_csv(os.path.join('data/raw/DatosW.csv'), delimiter=";", parse_dates=["Dates"], dayfirst=True).set_index('Dates')
cases_df['I_NV+PD'] = cases_df['InfectadosNoVacunados'].values+ cases_df['InfectadosPrimeraDosis'].values
cases_df['H_NV+PD'] = cases_df['HospitalizadosNoVacunados'].values + cases_df['HospitalizadosPrimeraDosis'].values
cases_df['M_NV+PD'] = cases_df['MuertesNoVacunados'].values + cases_df['MuertesPrimeraDosis'].values

# smooth cases, deaths and hospitalization.
cases_df = prepare_cases(cases_df, col='InfectadosNoVacunados')
cases_df = prepare_cases(cases_df, col='InfectadosPrimeraDosis')
cases_df = prepare_cases(cases_df, col='InfectadosEsquemaCompleto')
cases_df = prepare_cases(cases_df, col='InfectadosDosisRefuerzo')
cases_df = prepare_cases(cases_df, col='InfectadosTotal')
cases_df = prepare_cases(cases_df, col='HospitalizadosNoVacunados')
cases_df = prepare_cases(cases_df, col='HospitalizadosPrimeraDosis')
cases_df = prepare_cases(cases_df, col='HospitalizadosEsquemaCompleto')
cases_df = prepare_cases(cases_df, col='HospitalizadosDosisRefuerzo')
cases_df = prepare_cases(cases_df, col='HospitalizadosTotal')
cases_df = prepare_cases(cases_df, col='MuertesNoVacunados')
cases_df = prepare_cases(cases_df, col='MuertesPrimeraDosis')
cases_df = prepare_cases(cases_df, col='MuertesEsquemaCompleto')
cases_df = prepare_cases(cases_df, col='MuertesDosisRefuerzo')
cases_df = prepare_cases(cases_df, col='MuertesTotal')
cases_df = prepare_cases(cases_df, col='I_NV+PD')
cases_df = prepare_cases(cases_df, col='H_NV+PD')
cases_df = prepare_cases(cases_df, col='M_NV+PD')

# Calculate the infection fatality rate (IFR)
IFR_Verity = [ .00161, .00695, .0309, .0844, .161, .595, 1.93, 4.28, 7.80 ]
ifr_log    = np.log(IFR_Verity)

IFR = np.zeros((80))

ages_fit   = [9, 19, 29, 39, 49, 59, 69, 79, 89]
ifr_fitted = UnivariateSpline(ages_fit, ifr_log)

x = np.arange(0, 90, 1)
fitted_ifr = ifr_fitted(x)

ages_models = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
# Create IFR every 5 years
ifr = np.exp([fitted_ifr[0:6].mean(),fitted_ifr[6:11].mean(),fitted_ifr[11:16].mean(),fitted_ifr[16:21].mean(),
       fitted_ifr[21:26].mean(),fitted_ifr[27:31].mean(),fitted_ifr[31:36].mean(),fitted_ifr[37:41].mean(),
       fitted_ifr[41:46].mean(),fitted_ifr[47:51].mean(),fitted_ifr[51:56].mean(),fitted_ifr[56:61].mean(),
       fitted_ifr[61:66].mean(),fitted_ifr[67:71].mean(),fitted_ifr[71:76].mean(),fitted_ifr[76:].mean()])/100 #%

# define the infection fatality rate for vaccinated and unvaccinated people
ifr_m = np.mean(ifr)*3.85
iVfr  = 0.778*ifr_m

# define the range of the parameters
param_prior_dict  = {}
param_prior_dict["beta"]  = [1.02, 1.12] # Contact rate range
param_prior_dict["alpha"] = [0.29, 0.4] # Report rate range [1% - 100%]
param_prior_dict["Vr"] = [1, 1647460]
param_prior_dictV ={"Vr":[1131, 1331]}

# define the dates and parameters for the model.
date_init  = pd.to_datetime("2021-02-15")
date_end   = pd.to_datetime("2021-10-27")


lambda_inf    = 1.1
num_params    = 3
num_iters_mif = 360
alpha_mif     = 0.9 # Variance shrinking factor
num_ensembles = 200
N             = pop
num_state_vars   = 16
num_observations = 4
num_iters_save   = 4
num_save_iters   = int(num_iters_mif/num_iters_save)

# create the observation dataframe
obs_df = cases_df.copy()
obs_df  = obs_df.loc[date_init:date_end]
priors = np.load('outputs/samples_agg_Final_450.npz')
param_prior = priors['param_posterior']
x_prior1 = priors['x_posterior']

oev_df                     = pd.DataFrame(columns=["Dates", "OEV_confirmed", "OEV_deaths"])
oev_df["Dates"]            = obs_df.index.values
oev_df                     = oev_df.set_index("Dates")
oev_df["OEV_confirmed"]    = 1 + (0.2*obs_df["smoothed_I_NV+PD"].values)**2
oev_df["OEV_deaths"]       = 1 + (0.2*obs_df["smoothed_M_NV+PD"].values)**2
oev_df["OEV_confirmedV"]    = 1 + (0.2*obs_df["smoothed_InfectadosEsquemaCompleto"].values)**2
oev_df["OEV_deathsV"]       = 1 + (0.2*obs_df["smoothed_MuertesEsquemaCompleto"].values)**2
dates_assimilation = obs_df.index.get_level_values(0).values
dates = dates_assimilation
num_steps = len(dates)

######## INFERENCE  SETTINGS ########
# Range of parameters
param_range = np.array([v for k, v in param_prior_dict.items()])
std_param   = param_range[:,1]-param_range[:,0]
SIG         = std_param ** 2 / 4; #  initial covariance of parameters

num_steps     = len(obs_df) #387

obs_post_all = np.zeros((num_observations, num_ensembles, num_steps, num_iters_save))

para_post_all = np.zeros((num_params, num_ensembles, num_steps, num_iters_mif))
x_post_all    = np.zeros((num_state_vars, num_ensembles, num_steps, num_iters_save))
theta         = np.zeros((num_params, num_iters_mif+1))
################################################

# Run the model
print(f"Running MIF  \n")

cont = 0
cont1 = 0
for n in tqdm(range(num_iters_mif)): 
    if (n%num_save_iters)==0:
        obs_post_save = np.zeros((num_observations, num_ensembles, num_steps, num_save_iters))
        x_post_save = np.zeros((num_state_vars, num_ensembles, num_steps, num_save_iters))
        cont1 = 0
        
    if n==0:
        p_prior      = np.zeros((3,200))
        p_prior[:,:] = np.mean(param_prior.copy()[:,:,-1,:],-1)
        p_prior[2,:] = sample_params_uniform(param_prior_dictV, num_ensembles=num_ensembles)
        theta[:, n] = np.mean(p_prior, -1)
        x           = np.mean(x_prior1[:,:,-1,:],-1)

    else:
        p_prior     = np.zeros((3,200))
        params_mean = theta[:,n]
        params_var  = SIG * (alpha_mif**(n))**2
        p_prior[:,:] = np.mean(param_prior.copy()[:,:,-1,:],-1)
        p_prior[2,:]  = sample_params_normal(param_prior_dictV, [1231], [1000], num_ensembles=num_ensembles)
        x           = np.mean(x_prior1[:,:,-1,:],-1)

    param_post_time = np.zeros((num_params, num_ensembles, num_steps))
    x_post_time     = np.zeros((num_state_vars, num_ensembles, num_steps))
    obs_post_time   = np.zeros((num_observations, num_ensembles, num_steps))
    idx_date_update = 0

    confirmed_t      = np.zeros((num_ensembles, 1))
    deaths_t         = np.zeros((num_ensembles, 1))
    confirmedV_t     = np.zeros((num_ensembles, 1))
    deathsV_t        = np.zeros((num_ensembles, 1))

    st = time.time()
    for idx_t, date in enumerate(dates):

        beta  = p_prior[0, :]
        alpha = p_prior[1, :]
        Vr    = p_prior[2,:]
        x_ens = modelV(x, beta, ifr_m, iVfr, Vr, alpha, N)
        x     = x_ens

        confirmed_t      +=  np.expand_dims(x_ens[6,:], -1)
        deaths_t         +=  np.expand_dims(x_ens[7,:], -1)
        confirmedV_t     +=  np.expand_dims(x_ens[14,:], -1)
        deathsV_t        +=  np.expand_dims(x_ens[15,:], -1)

        # correct using daily observations
        if pd.to_datetime(date) == pd.to_datetime(dates_assimilation[idx_date_update]):

            x = inflate_ensembles(x, inflation_value=lambda_inf, num_ensembles=num_ensembles)
            x = checkbound_state_vars(x_state_ens=x, pop=N, num_params=num_params, num_ensembles=num_ensembles)

            p_prior = inflate_ensembles(p_prior, inflation_value=lambda_inf, num_ensembles=num_ensembles)
            p_prior = checkbound_params(param_prior_dict, p_prior, num_ensembles=num_ensembles)

            param_post = p_prior.copy()
            x_prior = x.copy()

            oev_deaths_time = oev_df.loc[date]["OEV_deaths"]
            deaths_time = obs_df.loc[date]["smoothed_M_NV+PD"]

            # Update parameters using confirmed deaths
            x_post, param_post, deaths_obs_post = eakf_step(x, param_post, np.squeeze(deaths_t), deaths_time, oev_deaths_time, param_prior_dict, num_var= num_state_vars)

            x_post     = checkbound_state_vars(x_state_ens=x_post, pop=N, num_params= num_state_vars, num_ensembles= num_ensembles)
            param_post = checkbound_params(param_prior_dict, params_ens=param_post, num_ensembles= num_ensembles)

            oev_deathsV_time = oev_df.loc[date]["OEV_deathsV"]
            deathsV_time = obs_df.loc[date]["smoothed_MuertesEsquemaCompleto"]

            # Update parameters using confirmed vaccinated deaths 
            x_post, param_post, deathsV_obs_post = eakf_step(x_post, param_post, np.squeeze(deathsV_t), deathsV_time, oev_deathsV_time, param_prior_dict, num_var= num_state_vars)

            x_post     = checkbound_state_vars(x_state_ens=x_post, pop=N, num_params= num_state_vars, num_ensembles= num_ensembles)
            param_post = checkbound_params(param_prior_dict, params_ens=param_post, num_ensembles= num_ensembles)


            oev_confirmed_time = oev_df.loc[date]["OEV_confirmed"]
            confirmed_time = obs_df.loc[date]["smoothed_I_NV+PD"]

            # Update parameters using confirmed cases
            x_post, param_post, confirmed_obs_post = eakf_step(x_post, param_post, np.squeeze(confirmed_t), confirmed_time, oev_confirmed_time, param_prior_dict, num_var=num_state_vars)

            x_post     = checkbound_state_vars(x_state_ens=x_post, pop=N, num_params=num_state_vars, num_ensembles=num_ensembles)
            param_post = checkbound_params(param_prior_dict, params_ens=param_post, num_ensembles=num_ensembles)


            oev_confirmedV_time = oev_df.loc[date]["OEV_confirmedV"]
            confirmedV_time = obs_df.loc[date]["smoothed_InfectadosEsquemaCompleto"]

            # Update parameters using confirmed vaccinated cases
            x_post, param_post, confirmedV_obs_post = eakf_step(x_post, param_post, np.squeeze(confirmedV_t), confirmedV_time, oev_confirmedV_time, param_prior_dict, num_var=num_state_vars)

            x_post     = checkbound_state_vars(x_state_ens=x_post, pop=N, num_params=num_state_vars, num_ensembles=num_ensembles)
            param_post = checkbound_params(param_prior_dict, params_ens=param_post, num_ensembles=num_ensembles)
            

            obs_post_time[0,:,idx_date_update]    = confirmed_obs_post
            obs_post_time[1,:,idx_date_update]    = deaths_obs_post
            obs_post_time[2,:,idx_date_update]    = confirmedV_obs_post
            obs_post_time[3,:,idx_date_update]    = deathsV_obs_post
            param_post_time[:,:,idx_date_update]  = param_post
            x_post_time[:,:,idx_date_update]      = x_post
            
            if idx_t == 0:
                param_post_time[:,:,idx_date_update]  = p_prior
                x_post_time[:,:,idx_date_update]      = x_prior
            
            x = x_post.copy()

            # Use posterior and next prior
            p_prior = param_post.copy()

            idx_date_update += 1

            confirmed_t    = np.zeros((num_ensembles, 1))
            deaths_t       = np.zeros((num_ensembles, 1))
            confirmedV_t    = np.zeros((num_ensembles, 1))
            deathsV_t       = np.zeros((num_ensembles, 1))


    et = time.time()
    
    x_post_save[:,:,:,cont1]   = x_post_time                                                     
    obs_post_save[:,:,:,cont1] = obs_post_time
    para_post_all[:,:,:,n] = param_post_time
    cont1 += 1
    # save the results
    if (n+1)%num_save_iters == 0 and n != 0:
        obs_post_all[:,:,:,cont] = obs_post_save.mean(-1)
        x_post_all[:,:,:,cont]       = x_post_save.mean(-1)
        cont += 1
        np.savez_compressed('outputs/samples_aggV_Final_{}.npz'.format(n+1),
                                param_posterior     = para_post_all,
                                x_posterior         = x_post_all,
                                obs_posterior       = obs_post_all,)

    theta[:,n+1] = param_post_time.mean(-1).mean(-1)
