# ------------------------------------------------------#
# Scientific computing
# ICTP/Serrapilheira 2022
# Final project: COVID19 Colombia
# First version 2022-08-15
# ------------------------------------------------------#

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from functions.adjust_cases import prepare_cases
from functions.utils_plotting import create_df_response

samples  = np.load('outputs/samples_agg_Final_450.npz')
samplesV = np.load('outputs/samples_aggV_Final_360.npz')

pop        = 40000000.0

# read cases by age without filtering
cases_df = pd.read_csv(os.path.join('data/raw/DatosW.csv'), delimiter=";", parse_dates=["Dates"], dayfirst=True).set_index('Dates')
cases_df['I_NV+PD'] = cases_df['InfectadosNoVacunados'].values+ cases_df['InfectadosPrimeraDosis'].values
cases_df['H_NV+PD'] = cases_df['HospitalizadosNoVacunados'].values + cases_df['HospitalizadosPrimeraDosis'].values
cases_df['M_NV+PD'] = cases_df['MuertesNoVacunados'].values + cases_df['MuertesPrimeraDosis'].values

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

date_init  = pd.to_datetime("2020-03-06")
date_end   = pd.to_datetime("2021-02-14")

date_initV  = pd.to_datetime("2021-02-15")
date_endV   = pd.to_datetime("2021-09-30")

obs_df = cases_df.copy()
obs_df  = obs_df.loc[date_init:date_end]

obs_dfV = cases_df.copy()
obs_dfV  = obs_dfV.loc[date_initV:date_endV]

oev_dfV                     = pd.DataFrame(columns=["Dates", "OEV_confirmedV", "OEV_deathsV"])
oev_dfV["Dates"]            = obs_dfV.index.values
oev_dfV                     = oev_dfV.set_index("Dates")
oev_dfV["OEV_confirmed"]    = np.maximum(1e-4, obs_dfV["smoothed_I_NV+PD"].values**2/100 )
oev_dfV["OEV_deaths"]       = np.maximum(25, obs_dfV["smoothed_M_NV+PD"].values**2/100 )
oev_dfV["OEV_confirmedV"]    = np.maximum(1e-4, obs_dfV["smoothed_InfectadosEsquemaCompleto"].values**2/100 )
oev_dfV["OEV_deathsV"]       = np.maximum(25, obs_dfV["smoothed_MuertesEsquemaCompleto"].values**2/100 )

dates_assimilationV = obs_dfV.index.get_level_values(0).values
datesV = dates_assimilationV
num_stepsV = len(datesV)

oev_df                     = pd.DataFrame(columns=["Dates", "OEV_confirmed", "OEV_deaths"])
oev_df["Dates"]            = obs_df.index.values
oev_df                     = oev_df.set_index("Dates")
oev_df["OEV_confirmed"]    = np.maximum(1e-4, obs_df["smoothed_InfectadosNoVacunados"].values**2/100 )
oev_df["OEV_deaths"]       = np.maximum(25, obs_df["smoothed_MuertesNoVacunados"].values**2/100 )

dates_assimilation = obs_df.index.get_level_values(0).values
dates = dates_assimilation
num_steps = len(dates)

param_post = samples['param_posterior']
param_postV = samplesV['param_posterior']

beta_post_all = param_post[0,:,:,:]
alpha_post_all = param_post[1,:,:,:]

beta_time   = np.mean(beta_post_all[:,:,:], -1)
alpha_time = np.mean(alpha_post_all[:,:,:], -1)

beta_post_allV = param_postV[0,:,:,:]
alpha_post_allV = param_postV[1,:,:,:]
Vr_post_allV = param_postV[2,:,:,:]

beta_timeV   = np.mean(beta_post_allV[:,:228,:], -1)
alpha_timeV = np.mean(alpha_post_allV[:,:228,:], -1)
Vr_timeV = np.mean(Vr_post_allV[:,:228,:], -1)

df_beta  = create_df_response(beta_time, time=num_steps, dates =dates_assimilation)
df_betaV  = create_df_response(beta_timeV, time=num_stepsV, dates =dates_assimilationV)
df_alpha = create_df_response(alpha_time, time=num_steps, dates =dates_assimilation)
df_alphaV = create_df_response(alpha_timeV, time=num_stepsV, dates =dates_assimilationV)

fig, ax = plt.subplots(2, 1, figsize=(12.5, 7.2), sharex=True)
ax[0].plot(df_beta.index.values, df_beta["median"], color='k', label='Median')
ax[0].fill_between(df_beta.index.values, df_beta["high_95"], df_beta["low_95"], color='c', alpha=0.3, label='95% CI')
ax[0].fill_between(df_beta.index.values, df_beta["high_50"], df_beta["low_50"], color='b', alpha=0.3, label='50% CI')

ax[0].plot(df_betaV.index.values, df_betaV["median"], color='k')#, label='Median')
ax[0].fill_between(df_betaV.index.values, df_betaV["high_95"], df_betaV["low_95"], color='c', alpha=0.3)#, label='95% CI')
ax[0].fill_between(df_betaV.index.values, df_betaV["high_50"], df_betaV["low_50"], color='b', alpha=0.3)#, label='50% CI')

ax[1].plot(df_alpha.index.values, df_alpha["median"], color='k', label='Median')
ax[1].fill_between(df_alpha.index.values, df_alpha["high_95"], df_alpha["low_95"], color='c', alpha=0.3, label='95% CI')
ax[1].fill_between(df_alpha.index.values, df_alpha["high_50"], df_alpha["low_50"], color='b', alpha=0.3, label='50% CI')

ax[1].plot(df_alphaV.index.values, df_alphaV["median"], color='k')#, label='Median')
ax[1].fill_between(df_alphaV.index.values, df_alphaV["high_95"], df_alphaV["low_95"], color='c', alpha=0.3)#, label='95% CI')
ax[1].fill_between(df_alphaV.index.values, df_alphaV["high_50"], df_alphaV["low_50"], color='b', alpha=0.3)#, label='50% CI')

ax[1].tick_params( which='both', axis='x', labelrotation=90, labelsize=15)
ax[0].tick_params( which='both', axis='y', labelsize=15)
ax[1].tick_params( which='both', axis='y', labelsize=15)
ax[0].set_ylabel(r"Contact rate | $\beta$ ", fontsize=15)
ax[0].axvline(x=pd.to_datetime('2021-02-15'), linestyle='--', color='r')

ax[1].set_ylabel(r"Report rate | $\alpha$", fontsize=15)
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels,fontsize=15, loc=[0.85,0.5])
ax[1].axvline(x=pd.to_datetime('2021-02-15'), linestyle='--', color='r')
plt.tight_layout()
plt.savefig("figs/inference_parameters.png")
plt.show()

df_Vr  = create_df_response(Vr_timeV, time=num_stepsV, dates =dates_assimilationV)

fig, ax = plt.subplots(1, 1, figsize=(12.5, 7.2), sharex=False)
ax.plot(df_Vr.index.values, df_Vr["median"], color='k', label='Median')
ax.fill_between(df_Vr.index.values, df_Vr["high_95"], df_Vr["low_95"], color='c', alpha=0.3, label='95% CI')
ax.fill_between(df_Vr.index.values, df_Vr["high_50"], df_Vr["low_50"], color='b', alpha=0.3, label='50% CI')

ax.set_ylabel(r"# Vaccines", fontsize=15)
ax.tick_params( which='both', axis='x', labelrotation=90, labelsize=15)
plt.savefig("figs/inference_vaccine.png")
plt.show()


obs_all = samples['obs_posterior']
obs_allV = samplesV['obs_posterior']

confirmed_post_all   = np.mean(obs_all[0,:,:,:],-1)
deaths_post_all      = np.mean(obs_all[1,:,:,:],-1)
confirmed_post_allV  = np.mean(obs_allV[0,:,:228,:],-1)
deaths_post_allV     = np.mean(obs_allV[1,:,:228,:],-1)
confirmedVV_post_all = np.mean(obs_allV[2,:,:228,:],-1)
deathsVV_post_all    = np.mean(obs_allV[3,:,:228,:],-1)
df_confirmed   = create_df_response(confirmed_post_all, time=num_steps, dates =dates_assimilation)
df_deaths      = create_df_response(deaths_post_all, time=num_steps, dates =dates_assimilation)
df_confirmedV  = create_df_response(confirmed_post_allV, time=num_stepsV, dates =dates_assimilationV)
df_deathsV     = create_df_response(deaths_post_allV, time=num_stepsV, dates =dates_assimilationV)
df_confirmedVV = create_df_response(confirmedVV_post_all, time=num_stepsV, dates =dates_assimilationV)
df_deathsVV    = create_df_response(deathsVV_post_all, time=num_stepsV, dates =dates_assimilationV)

fig, ax = plt.subplots(2, 2, figsize=(17.5, 10), sharex=True)

ax[0,0].plot(df_confirmedV.index.values, df_confirmedV["median"], color='gray')#, label='Median')
ax[0,0].fill_between(df_confirmedV.index.values, df_confirmedV["high_95"], df_confirmedV["low_95"], color='cyan', alpha=0.3)#, label='95% CI')
ax[0,0].fill_between(df_confirmedV.index.values, df_confirmedV["high_50"], df_confirmedV["low_50"], color='blue', alpha=0.3)#, label='50% CI')
ax[0,0].scatter(obs_dfV.index.values, obs_dfV["smoothed_I_NV+PD"], edgecolors="w", facecolor="darkred")

ax[0,0].plot(df_confirmed.index.values, df_confirmed["median"], color='gray')#, label='Median')
ax[0,0].fill_between(df_confirmed.index.values, df_confirmed["high_95"], df_confirmed["low_95"], color='cyan', alpha=0.3)#, label='95% CI')
ax[0,0].fill_between(df_confirmed.index.values, df_confirmed["high_50"], df_confirmed["low_50"], color='blue', alpha=0.3)#, label='50% CI')
ax[0,0].scatter(obs_df.index.values, obs_df["smoothed_InfectadosNoVacunados"], edgecolors="w", facecolor="darkred")

ax[0,1].plot(df_deathsV.index.values, df_deathsV["median"], color='gray', label='Median')
ax[0,1].fill_between(df_deathsV.index.values, df_deathsV["high_95"], df_deathsV["low_95"], color='cyan', alpha=0.3)#, label='95% CI')
ax[0,1].fill_between(df_deathsV.index.values, df_deathsV["high_50"], df_deathsV["low_50"], color='blue', alpha=0.3)#, label='50% CI')
ax[0,1].scatter(obs_dfV.index.values, obs_dfV["smoothed_M_NV+PD"], edgecolors="w", facecolor="darkred")

ax[0,1].plot(df_deaths.index.values, df_deaths["median"], color='gray')#, label='Median')
ax[0,1].fill_between(df_deaths.index.values, df_deaths["high_95"], df_deaths["low_95"], color='cyan', alpha=0.3)#, label='95% CI')
ax[0,1].fill_between(df_deaths.index.values, df_deaths["high_50"], df_deaths["low_50"], color='blue', alpha=0.3)#, label='50% CI')
ax[0,1].scatter(obs_df.index.values, obs_df["smoothed_MuertesNoVacunados"], edgecolors="w", facecolor="darkred")

ax[1,0].plot(df_confirmedVV.index.values, df_confirmedVV["median"], color='gray')#, label='Median')
ax[1,0].fill_between(df_confirmedVV.index.values, df_confirmedVV["high_95"], df_confirmedVV["low_95"], color='cyan', alpha=0.3)#, label='95% CI')
ax[1,0].fill_between(df_confirmedVV.index.values, df_confirmedVV["high_50"], df_confirmedVV["low_50"], color='blue', alpha=0.3)#, label='50% CI')
ax[1,0].scatter(obs_dfV.index.values, obs_dfV["smoothed_InfectadosEsquemaCompleto"], edgecolors="w", facecolor="darkred")

ax[1,1].plot(df_deathsVV.index.values, df_deathsVV["median"], color='gray', label='Median')
ax[1,1].fill_between(df_deathsVV.index.values, df_deathsVV["high_95"], df_deathsVV["low_95"], color='cyan', alpha=0.3, label='95% CI')
ax[1,1].fill_between(df_deathsVV.index.values, df_deathsVV["high_50"], df_deathsVV["low_50"], color='blue', alpha=0.3, label='50% CI')
ax[1,1].scatter(obs_dfV.index.values, obs_dfV["smoothed_MuertesEsquemaCompleto"], edgecolors="w", facecolor="darkred", label='Observations')

for axi in ax.flatten():
    axi.xaxis.set_major_locator(mdates.MonthLocator())
    axi.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
    axi.xaxis.set_minor_locator(mdates.WeekdayLocator())
    axi.tick_params( which='both', axis='both', labelsize=15)
    axi.tick_params( which='both', axis='x', rotation=90, labelsize=15)

    axi.spines['right'].set_visible(False)
    axi.spines['top'].set_visible(False)

ax[0,0].axvline(x=pd.to_datetime('2021-02-15'), linestyle='--', color='r')
ax[0,1].axvline(x=pd.to_datetime('2021-02-15'), linestyle='--', color='r')
ax[1,0].axvline(x=pd.to_datetime('2021-02-15'), linestyle='--', color='r')
ax[1,1].axvline(x=pd.to_datetime('2021-02-15'), linestyle='--', color='r')
ax[0,0].set_ylabel("Confirmed Cases", fontsize=15)
ax[0,1].set_ylabel("Deaths", fontsize=15)
ax[1,0].set_ylabel("Confirmed Vaccinated Cases", fontsize=15)
ax[1,1].set_ylabel("Deaths Vaccinated", fontsize=15)
ax[1,1].legend(fontsize=14, loc=(1.04,1))

plt.savefig("figs/fitting_model.png")
plt.show()

x_post_all = samples['x_posterior']
x_post_allV = samplesV['x_posterior']

S_time = np.mean(x_post_all[0,:,:,:6], -1)
E_time = np.mean(x_post_all[1,:,:,:6], -1)
Ir_time = np.mean(x_post_all[2,:,:,:6], -1)
Iu_time = np.mean(x_post_all[3,:,:,:6], -1)
Ih_time = np.mean(x_post_all[4,:,:,:6], -1)
R_time = np.mean(x_post_all[5,:,:,:6], -1)
C_time = np.mean(x_post_all[6,:,:,:6], -1)
D_time = np.mean(x_post_all[7,:,:,:6], -1)

df_S  = create_df_response(S_time, time=num_steps, dates =dates_assimilation)
df_E  = create_df_response(E_time, time=num_steps, dates =dates_assimilation)
df_Ir  = create_df_response(Ir_time, time=num_steps, dates =dates_assimilation)
df_Iu  = create_df_response(Iu_time, time=num_steps, dates =dates_assimilation)
df_Ih  = create_df_response(Ih_time, time=num_steps, dates =dates_assimilation)
df_R  = create_df_response(R_time, time=num_steps, dates =dates_assimilation)
df_C  = create_df_response(C_time, time=num_steps, dates =dates_assimilation)
df_D  = create_df_response(D_time, time=num_steps, dates =dates_assimilation)

#---------------------------------------------------------------

fig, ax = plt.subplots(2, 3, figsize=(25, 21.6))

ax[0,0].plot(df_S.index.values, df_S["median"], color='k')#, label='Median')
ax[0,0].fill_between(df_S.index.values, df_S["high_95"], df_S["low_95"], color='k', alpha=0.3)#, label='95% CI')
ax[0,0].fill_between(df_S.index.values, df_S["high_50"], df_S["low_50"], color='k', alpha=0.3)#, label='50% CI')

ax[0,0].xaxis.set_major_locator(mdates.MonthLocator())
ax[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
ax[0,0].xaxis.set_minor_locator(mdates.WeekdayLocator())
ax[0,0].tick_params( which='both', axis='both', labelsize=15)
ax[0,0].spines['right'].set_visible(False)
ax[0,0].spines['top'].set_visible(False)

ax[0,0].tick_params( which='both', axis='x', labelrotation=90, labelsize=15)
ax[0,0].set_xlabel("Date", fontsize=15)
ax[0,0].set_ylabel("Susceptible", fontsize=15)
ax[0,0].set_title("Susceptible", fontsize=15)
plt.tight_layout()

#---------------------------------------------------------------

ax[0,1].plot(df_E.index.values, df_E["median"], color='k')#, label='Median')
ax[0,1].fill_between(df_E.index.values, df_E["high_95"], df_E["low_95"], color='k', alpha=0.3)#, label='95% CI')
ax[0,1].fill_between(df_E.index.values, df_E["high_50"], df_E["low_50"], color='k', alpha=0.3)#, label='50% CI')

ax[0,1].xaxis.set_major_locator(mdates.MonthLocator())
ax[0,1].xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
ax[0,1].xaxis.set_minor_locator(mdates.WeekdayLocator())
ax[0,1].tick_params( which='both', axis='both', labelsize=15)
ax[0,1].spines['right'].set_visible(False)
ax[0,1].spines['top'].set_visible(False)

ax[0,1].tick_params( which='both', axis='x', labelrotation=90, labelsize=15)
ax[0,1].set_xlabel("Date", fontsize=15)
ax[0,1].set_ylabel("Exposed", fontsize=15)
ax[0,1].set_title("Exposed", fontsize=15)
plt.tight_layout()

#---------------------------------------------------------------

ax[0,2].plot(df_Ir.index.values, df_Ir["median"], color='k', label='Median')
ax[0,2].fill_between(df_Ir.index.values, df_Ir["high_95"], df_Ir["low_95"], color='k', alpha=0.3, label='95% CI')
ax[0,2].fill_between(df_Ir.index.values, df_Ir["high_50"], df_Ir["low_50"], color='k', alpha=0.3, label='50% CI')

ax[0,2].xaxis.set_major_locator(mdates.MonthLocator())
ax[0,2].xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
ax[0,2].xaxis.set_minor_locator(mdates.WeekdayLocator())
ax[0,2].tick_params( which='both', axis='both', labelsize=15)
ax[0,2].spines['right'].set_visible(False)
ax[0,2].spines['top'].set_visible(False)

ax[0,2].tick_params( which='both', axis='x', labelrotation=90, labelsize=15)
ax[0,2].set_xlabel("Date", fontsize=15)
ax[0,2].set_ylabel("Infected reported", fontsize=15)
ax[0,2].set_title("Infected reported", fontsize=15)
ax[0,2].legend(fontsize=15, loc='best')
plt.tight_layout()

#---------------------------------------------------------------

ax[1,0].plot(df_Iu.index.values, df_Iu["median"], color='k')#, label='Median')
ax[1,0].fill_between(df_Iu.index.values, df_Iu["high_95"], df_Iu["low_95"], color='k', alpha=0.3)#, label='95% CI')
ax[1,0].fill_between(df_Iu.index.values, df_Iu["high_50"], df_Iu["low_50"], color='k', alpha=0.3)#, label='50% CI')

ax[1,0].xaxis.set_major_locator(mdates.MonthLocator())
ax[1,0].xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
ax[1,0].xaxis.set_minor_locator(mdates.WeekdayLocator())
ax[1,0].tick_params( which='both', axis='both', labelsize=15)
ax[1,0].spines['right'].set_visible(False)
ax[1,0].spines['top'].set_visible(False)

ax[1,0].tick_params( which='both', axis='x', labelrotation=90, labelsize=15)
ax[1,0].set_xlabel("Date", fontsize=15)
ax[1,0].set_ylabel("Infected unreported", fontsize=15)
ax[1,0].set_title("Infected unreported", fontsize=15)
plt.tight_layout()

#---------------------------------------------------------------

ax[1,1].plot(df_Ih.index.values, df_Ih["median"], color='k')#, label='Median')
ax[1,1].fill_between(df_Ih.index.values, df_Ih["high_95"], df_Ih["low_95"], color='k', alpha=0.3)#, label='95% CI')
ax[1,1].fill_between(df_Ih.index.values, df_Ih["high_50"], df_Ih["low_50"], color='k', alpha=0.3)#, label='50% CI')

ax[1,1].xaxis.set_major_locator(mdates.MonthLocator())
ax[1,1].xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
ax[1,1].xaxis.set_minor_locator(mdates.WeekdayLocator())
ax[1,1].tick_params( which='both', axis='both', labelsize=15)
ax[1,1].spines['right'].set_visible(False)
ax[1,1].spines['top'].set_visible(False)

ax[1,1].tick_params( which='both', axis='x', labelrotation=90, labelsize=15)
ax[1,1].set_xlabel("Date", fontsize=15)
ax[1,1].set_ylabel("Deaths", fontsize=15)
ax[1,1].set_title("Deaths", fontsize=15)
plt.tight_layout()

#---------------------------------------------------------------

ax[1,2].plot(df_R.index.values, df_R["median"], color='k')#, label='Median')
ax[1,2].fill_between(df_R.index.values, df_R["high_95"], df_R["low_95"], color='k', alpha=0.3)#, label='95% CI')
ax[1,2].fill_between(df_R.index.values, df_R["high_50"], df_R["low_50"], color='k', alpha=0.3)#, label='50% CI')

ax[1,2].xaxis.set_major_locator(mdates.MonthLocator())
ax[1,2].xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
ax[1,2].xaxis.set_minor_locator(mdates.WeekdayLocator())
ax[1,2].tick_params( which='both', axis='both', labelsize=15)
ax[1,2].spines['right'].set_visible(False)
ax[1,2].spines['top'].set_visible(False)

ax[1,2].tick_params( which='both', axis='x', labelrotation=90, labelsize=15)
ax[1,2].set_xlabel("Date", fontsize=15)
ax[1,2].set_ylabel("Recovered", fontsize=15)
ax[1,2].set_title("Recovered", fontsize=15)
plt.tight_layout()

S_time = np.mean(x_post_allV[0,:,:228,:6], -1)
E_time = np.mean(x_post_allV[1,:,:228,:6], -1)
Ir_time = np.mean(x_post_allV[2,:,:228,:6], -1)
Iu_time = np.mean(x_post_allV[3,:,:228,:6], -1)
Ih_time = np.mean(x_post_allV[4,:,:228,:6], -1)
R_time = np.mean(x_post_allV[5,:,:228,:6], -1)
# H_time = np.mean(x_post_allV[6,:,:228,:6], -1)
C_time = np.mean(x_post_allV[6,:,:228,:6], -1)
D_time = np.mean(x_post_allV[7,:,:228,:6], -1)

df_S  = create_df_response(S_time, time=num_stepsV, dates =dates_assimilationV)
df_E  = create_df_response(E_time, time=num_stepsV, dates =dates_assimilationV)
df_Ir  = create_df_response(Ir_time, time=num_stepsV, dates =dates_assimilationV)
df_Iu  = create_df_response(Iu_time, time=num_stepsV, dates =dates_assimilationV)
df_Ih  = create_df_response(Ih_time, time=num_stepsV, dates =dates_assimilationV)
df_R  = create_df_response(R_time, time=num_stepsV, dates =dates_assimilationV)
df_H  = create_df_response(C_time, time=num_stepsV, dates =dates_assimilationV)
df_C  = create_df_response(C_time, time=num_stepsV, dates =dates_assimilationV)
df_D  = create_df_response(D_time, time=num_stepsV, dates =dates_assimilationV)

#---------------------------------------------------------------

# fig, ax = plt.subplots(2, 3, figsize=(25, 21.6))

ax[0,0].plot(df_S.index.values, df_S["median"], color='k')#, label='Median')
ax[0,0].fill_between(df_S.index.values, df_S["high_95"], df_S["low_95"], color='k', alpha=0.3)#, label='95% CI')
ax[0,0].fill_between(df_S.index.values, df_S["high_50"], df_S["low_50"], color='k', alpha=0.3)#, label='50% CI')

#sns.barplot(data=df_response, x='date', y='valuye')

ax[0,0].xaxis.set_major_locator(mdates.MonthLocator())
ax[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
ax[0,0].xaxis.set_minor_locator(mdates.WeekdayLocator())
ax[0,0].tick_params( which='both', axis='both', labelsize=15)
ax[0,0].spines['right'].set_visible(False)
ax[0,0].spines['top'].set_visible(False)

ax[0,0].tick_params( which='both', axis='x', labelrotation=90, labelsize=15)
ax[0,0].set_xlabel("Date", fontsize=15)
ax[0,0].set_ylabel("Susceptible", fontsize=15)
ax[0,0].set_title("Susceptible", fontsize=15)
# ax[0,0].legend(fontsize=15, loc='upper right')
plt.tight_layout()

#---------------------------------------------------------------

ax[0,1].plot(df_E.index.values, df_E["median"], color='k')#, label='Median')
ax[0,1].fill_between(df_E.index.values, df_E["high_95"], df_E["low_95"], color='k', alpha=0.3)#, label='95% CI')
ax[0,1].fill_between(df_E.index.values, df_E["high_50"], df_E["low_50"], color='k', alpha=0.3)#, label='50% CI')

#sns.barplot(data=df_response, x='date', y='valuye')

ax[0,1].xaxis.set_major_locator(mdates.MonthLocator())
ax[0,1].xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
ax[0,1].xaxis.set_minor_locator(mdates.WeekdayLocator())
ax[0,1].tick_params( which='both', axis='both', labelsize=15)
ax[0,1].spines['right'].set_visible(False)
ax[0,1].spines['top'].set_visible(False)

ax[0,1].tick_params( which='both', axis='x', labelrotation=90, labelsize=15)
ax[0,1].set_xlabel("Date", fontsize=15)
ax[0,1].set_ylabel("Exposed", fontsize=15)
ax[0,1].set_title("Exposed", fontsize=15)
# ax[0,1].legend(fontsize=15, loc='upper right')
plt.tight_layout()

#---------------------------------------------------------------

ax[0,2].plot(df_Ir.index.values, df_Ir["median"], color='k')#, label='Median')
ax[0,2].fill_between(df_Ir.index.values, df_Ir["high_95"], df_Ir["low_95"], color='k', alpha=0.3)#, label='95% CI')
ax[0,2].fill_between(df_Ir.index.values, df_Ir["high_50"], df_Ir["low_50"], color='k', alpha=0.3)#, label='50% CI')

#sns.barplot(data=df_response, x='date', y='valuye')

ax[0,2].xaxis.set_major_locator(mdates.MonthLocator())
ax[0,2].xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
ax[0,2].xaxis.set_minor_locator(mdates.WeekdayLocator())
ax[0,2].tick_params( which='both', axis='both', labelsize=15)
ax[0,2].spines['right'].set_visible(False)
ax[0,2].spines['top'].set_visible(False)

ax[0,2].tick_params( which='both', axis='x', labelrotation=90, labelsize=15)
ax[0,2].set_xlabel("Date", fontsize=15)
ax[0,2].set_ylabel("Infected reported", fontsize=15)
ax[0,2].set_title("Infected reported", fontsize=15)
# ax[0,2].legend(fontsize=15, loc='upper right')
plt.tight_layout()

#---------------------------------------------------------------

ax[1,0].plot(df_Iu.index.values, df_Iu["median"], color='k')#, label='Median')
ax[1,0].fill_between(df_Iu.index.values, df_Iu["high_95"], df_Iu["low_95"], color='k', alpha=0.3)#, label='95% CI')
ax[1,0].fill_between(df_Iu.index.values, df_Iu["high_50"], df_Iu["low_50"], color='k', alpha=0.3)#, label='50% CI')

#sns.barplot(data=df_response, x='date', y='valuye')

ax[1,0].xaxis.set_major_locator(mdates.MonthLocator())
ax[1,0].xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
ax[1,0].xaxis.set_minor_locator(mdates.WeekdayLocator())
ax[1,0].tick_params( which='both', axis='both', labelsize=15)
ax[1,0].spines['right'].set_visible(False)
ax[1,0].spines['top'].set_visible(False)

ax[1,0].tick_params( which='both', axis='x', labelrotation=90, labelsize=15)
ax[1,0].set_xlabel("Date", fontsize=15)
ax[1,0].set_ylabel("Infected unreported", fontsize=15)
ax[1,0].set_title("Infected unreported", fontsize=15)
# ax[1,0].legend(fontsize=15, loc='upper right')
plt.tight_layout()

#---------------------------------------------------------------

ax[1,1].plot(df_Ih.index.values, df_Ih["median"], color='k')#, label='Median')
ax[1,1].fill_between(df_Ih.index.values, df_Ih["high_95"], df_Ih["low_95"], color='k', alpha=0.3)#, label='95% CI')
ax[1,1].fill_between(df_Ih.index.values, df_Ih["high_50"], df_Ih["low_50"], color='k', alpha=0.3)#, label='50% CI')

#sns.barplot(data=df_response, x='date', y='valuye')

ax[1,1].xaxis.set_major_locator(mdates.MonthLocator())
ax[1,1].xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
ax[1,1].xaxis.set_minor_locator(mdates.WeekdayLocator())
ax[1,1].tick_params( which='both', axis='both', labelsize=15)
ax[1,1].spines['right'].set_visible(False)
ax[1,1].spines['top'].set_visible(False)

ax[1,1].tick_params( which='both', axis='x', labelrotation=90, labelsize=15)
ax[1,1].set_xlabel("Date", fontsize=15)
ax[1,1].set_ylabel("Deaths", fontsize=15)
ax[1,1].set_title("Deaths", fontsize=15)
# ax[1,1].legend(fontsize=15, loc='upper right')
plt.tight_layout()

#---------------------------------------------------------------

ax[1,2].plot(df_R.index.values, df_R["median"], color='k')#, label='Median')
ax[1,2].fill_between(df_R.index.values, df_R["high_95"], df_R["low_95"], color='k', alpha=0.3)#, label='95% CI')
ax[1,2].fill_between(df_R.index.values, df_R["high_50"], df_R["low_50"], color='k', alpha=0.3)#, label='50% CI')

#sns.barplot(data=df_response, x='date', y='valuye')

ax[1,2].xaxis.set_major_locator(mdates.MonthLocator())
ax[1,2].xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
ax[1,2].xaxis.set_minor_locator(mdates.WeekdayLocator())
ax[1,2].tick_params( which='both', axis='both', labelsize=15)
ax[1,2].spines['right'].set_visible(False)
ax[1,2].spines['top'].set_visible(False)

ax[1,2].tick_params( which='both', axis='x', labelrotation=90, labelsize=15)
ax[1,2].set_xlabel("Date", fontsize=15)
ax[1,2].set_ylabel("Recovered", fontsize=15)
ax[1,2].set_title("Recovered", fontsize=15)
# ax[1,2].legend(fontsize=15, loc='upper right')
plt.tight_layout()

ax[0,0].axvline(x=pd.to_datetime('2021-02-15'), linestyle='--', color='r')
ax[0,1].axvline(x=pd.to_datetime('2021-02-15'), linestyle='--', color='r')
ax[1,0].axvline(x=pd.to_datetime('2021-02-15'), linestyle='--', color='r')
ax[1,1].axvline(x=pd.to_datetime('2021-02-15'), linestyle='--', color='r')
ax[0,2].axvline(x=pd.to_datetime('2021-02-15'), linestyle='--', color='r')
ax[1,2].axvline(x=pd.to_datetime('2021-02-15'), linestyle='--', color='r')

plt.savefig("figs/state_variables_unvaccines.png")
plt.show()

S_time = np.mean(x_post_allV[8,:,:228,:6], -1)
E_time = np.mean(x_post_allV[9,:,:228,:6], -1)
Ir_time = np.mean(x_post_allV[10,:,:228,:6], -1)
Iu_time = np.mean(x_post_allV[11,:,:228,:6], -1)
Ih_time = np.mean(x_post_allV[12,:,:228,:6], -1)
R_time = np.mean(x_post_allV[13,:,:228,:6], -1)
# H_time = np.mean(x_post_allV[15,:,:228,:6], -1)
C_time = np.mean(x_post_allV[14,:,:228,:6], -1)
D_time = np.mean(x_post_allV[15,:,:228,:6], -1)

df_S  = create_df_response(S_time, time=num_stepsV, dates =dates_assimilationV)
df_E  = create_df_response(E_time, time=num_stepsV, dates =dates_assimilationV)
df_Ir  = create_df_response(Ir_time, time=num_stepsV, dates =dates_assimilationV)
df_Iu  = create_df_response(Iu_time, time=num_stepsV, dates =dates_assimilationV)
df_Ih  = create_df_response(Ih_time, time=num_stepsV, dates =dates_assimilationV)
df_R  = create_df_response(R_time, time=num_stepsV, dates =dates_assimilationV)
df_H  = create_df_response(C_time, time=num_stepsV, dates =dates_assimilationV)
df_C  = create_df_response(C_time, time=num_stepsV, dates =dates_assimilationV)
df_D  = create_df_response(D_time, time=num_stepsV, dates =dates_assimilationV)

#---------------------------------------------------------------

fig, ax = plt.subplots(2, 3, figsize=(25, 21.6))

ax[0,0].plot(df_S.index.values, df_S["median"], color='b')#, label='Median')
ax[0,0].fill_between(df_S.index.values, df_S["high_95"], df_S["low_95"], color='b', alpha=0.3)#, label='95% CI')
ax[0,0].fill_between(df_S.index.values, df_S["high_50"], df_S["low_50"], color='b', alpha=0.3)#, label='50% CI')

#sns.barplot(data=df_response, x='date', y='valuye')

ax[0,0].xaxis.set_major_locator(mdates.MonthLocator())
ax[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
ax[0,0].xaxis.set_minor_locator(mdates.WeekdayLocator())
ax[0,0].tick_params( which='both', axis='both', labelsize=15)
ax[0,0].spines['right'].set_visible(False)
ax[0,0].spines['top'].set_visible(False)

ax[0,0].tick_params( which='both', axis='x', labelrotation=90, labelsize=15)
ax[0,0].set_xlabel("Date", fontsize=15)
ax[0,0].set_ylabel("Susceptible", fontsize=15)
ax[0,0].set_title("SusceptibleV", fontsize=15)
# ax[0,0].legend(fontsize=15, loc='upper right')
plt.tight_layout()

#---------------------------------------------------------------

ax[0,1].plot(df_E.index.values, df_E["median"], color='b')#, label='Median')
ax[0,1].fill_between(df_E.index.values, df_E["high_95"], df_E["low_95"], color='b', alpha=0.3)#, label='95% CI')
ax[0,1].fill_between(df_E.index.values, df_E["high_50"], df_E["low_50"], color='b', alpha=0.3)#, label='50% CI')

#sns.barplot(data=df_response, x='date', y='valuye')

ax[0,1].xaxis.set_major_locator(mdates.MonthLocator())
ax[0,1].xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
ax[0,1].xaxis.set_minor_locator(mdates.WeekdayLocator())
ax[0,1].tick_params( which='both', axis='both', labelsize=15)
ax[0,1].spines['right'].set_visible(False)
ax[0,1].spines['top'].set_visible(False)

ax[0,1].tick_params( which='both', axis='x', labelrotation=90, labelsize=15)
ax[0,1].set_xlabel("Date", fontsize=15)
ax[0,1].set_ylabel("Exposed", fontsize=15)
ax[0,1].set_title("ExposedV", fontsize=15)
# ax[0,1].legend(fontsize=15, loc='upper right')
plt.tight_layout()

#---------------------------------------------------------------

ax[0,2].plot(df_Ir.index.values, df_Ir["median"], color='b', label='Median vaccinated')
ax[0,2].fill_between(df_Ir.index.values, df_Ir["high_95"], df_Ir["low_95"], color='b', alpha=0.3)#)#, label='95% CI')
ax[0,2].fill_between(df_Ir.index.values, df_Ir["high_50"], df_Ir["low_50"], color='b', alpha=0.3)#)#, label='50% CI')

#sns.barplot(data=df_response, x='date', y='valuye')

ax[0,2].xaxis.set_major_locator(mdates.MonthLocator())
ax[0,2].xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
ax[0,2].xaxis.set_minor_locator(mdates.WeekdayLocator())
ax[0,2].tick_params( which='both', axis='both', labelsize=15)
ax[0,2].spines['right'].set_visible(False)
ax[0,2].spines['top'].set_visible(False)

ax[0,2].tick_params( which='both', axis='x', labelrotation=90, labelsize=15)
ax[0,2].set_xlabel("Date", fontsize=15)
ax[0,2].set_ylabel("Infected reported", fontsize=15)
ax[0,2].set_title("Infected reportedV", fontsize=15)
ax[0,2].legend(fontsize=15, loc='best')
plt.tight_layout()

#---------------------------------------------------------------

ax[1,0].plot(df_Iu.index.values, df_Iu["median"], color='b')#, label='Median')
ax[1,0].fill_between(df_Iu.index.values, df_Iu["high_95"], df_Iu["low_95"], color='b', alpha=0.3)#, label='95% CI')
ax[1,0].fill_between(df_Iu.index.values, df_Iu["high_50"], df_Iu["low_50"], color='b', alpha=0.3)#, label='50% CI')

#sns.barplot(data=df_response, x='date', y='valuye')

ax[1,0].xaxis.set_major_locator(mdates.MonthLocator())
ax[1,0].xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
ax[1,0].xaxis.set_minor_locator(mdates.WeekdayLocator())
ax[1,0].tick_params( which='both', axis='both', labelsize=15)
ax[1,0].spines['right'].set_visible(False)
ax[1,0].spines['top'].set_visible(False)

ax[1,0].tick_params( which='both', axis='x', labelrotation=90, labelsize=15)
ax[1,0].set_xlabel("Date", fontsize=15)
ax[1,0].set_ylabel("Infected unreported", fontsize=15)
ax[1,0].set_title("Infected unreportedV", fontsize=15)
# ax[1,0].legend(fontsize=15, loc='upper right')
plt.tight_layout()

#---------------------------------------------------------------

ax[1,1].plot(df_Ih.index.values, df_Ih["median"], color='b')#, label='Median')
ax[1,1].fill_between(df_Ih.index.values, df_Ih["high_95"], df_Ih["low_95"], color='b', alpha=0.3)#, label='95% CI')
ax[1,1].fill_between(df_Ih.index.values, df_Ih["high_50"], df_Ih["low_50"], color='b', alpha=0.3)#, label='50% CI')

#sns.barplot(data=df_response, x='date', y='valuye')

ax[1,1].xaxis.set_major_locator(mdates.MonthLocator())
ax[1,1].xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
ax[1,1].xaxis.set_minor_locator(mdates.WeekdayLocator())
ax[1,1].tick_params( which='both', axis='both', labelsize=15)
ax[1,1].spines['right'].set_visible(False)
ax[1,1].spines['top'].set_visible(False)

ax[1,1].tick_params( which='both', axis='x', labelrotation=90, labelsize=15)
ax[1,1].set_xlabel("Date", fontsize=15)
ax[1,1].set_ylabel("Deaths", fontsize=15)
ax[1,1].set_title("DeathsV", fontsize=15)
# ax[1,1].legend(fontsize=15, loc='upper right')
plt.tight_layout()

#---------------------------------------------------------------

ax[1,2].plot(df_R.index.values, df_R["median"], color='b')#, label='Median')
ax[1,2].fill_between(df_R.index.values, df_R["high_95"], df_R["low_95"], color='b', alpha=0.3)#, label='95% CI')
ax[1,2].fill_between(df_R.index.values, df_R["high_50"], df_R["low_50"], color='b', alpha=0.3)#, label='50% CI')

#sns.barplot(data=df_response, x='date', y='valuye')

ax[1,2].xaxis.set_major_locator(mdates.MonthLocator())
ax[1,2].xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
ax[1,2].xaxis.set_minor_locator(mdates.WeekdayLocator())
ax[1,2].tick_params( which='both', axis='both', labelsize=15)
ax[1,2].spines['right'].set_visible(False)
ax[1,2].spines['top'].set_visible(False)

ax[1,2].tick_params( which='both', axis='x', labelrotation=90, labelsize=15)
ax[1,2].set_xlabel("Date", fontsize=15)
ax[1,2].set_ylabel("Recovered", fontsize=15)
ax[1,2].set_title("RecoveredV", fontsize=15)
# ax[1,2].legend(fontsize=15, loc='upper right')
plt.tight_layout()

plt.savefig("figs/state_variables_vaccines.png")
plt.show()
