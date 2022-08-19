# ------------------------------------------------------#
# Scientific computing
# ICTP/Serrapilheira 2022
# Final project: COVID19 Colombia
# First version 2022-08-15
# ------------------------------------------------------#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from functions.utils_plotting import create_df_response

samples = np.load('outputs/samples_agg_Final_450.npz')

param_post = samples['param_posterior']

beta_post_all = param_post[0,:,:,:]
alpha_post_all = param_post[1,:,:,:]

beta_time   = np.mean(beta_post_all[:,:,:], -1)
alpha_time = np.mean(alpha_post_all[:,:,:], -1)

df_beta  = create_df_response(beta_time, time=num_steps, dates =dates_assimilation)
df_alpha = create_df_response(alpha_time, time=num_steps, dates =dates_assimilation)

fig, ax = plt.subplots(2, 1, figsize=(12.5, 7.2), sharex=True)
ax[0].plot(df_beta.index.values, df_beta["median"], color='k', label='Median')
ax[0].fill_between(df_beta.index.values, df_beta["high_95"], df_beta["low_95"], color='c', alpha=0.3, label='95% CI')
ax[0].fill_between(df_beta.index.values, df_beta["high_50"], df_beta["low_50"], color='b', alpha=0.3, label='50% CI')

ax[1].plot(df_alpha.index.values, df_alpha["median"], color='k', label='Median')
ax[1].fill_between(df_alpha.index.values, df_alpha["high_95"], df_alpha["low_95"], color='c', alpha=0.3, label='95% CI')
ax[1].fill_between(df_alpha.index.values, df_alpha["high_50"], df_alpha["low_50"], color='b', alpha=0.3, label='50% CI')

ax[1].tick_params( which='both', axis='x', labelrotation=90, labelsize=15)
ax[0].tick_params( which='both', axis='y', labelsize=15)
ax[1].tick_params( which='both', axis='y', labelsize=15)
ax[0].set_ylabel(r"Contact rate | $\beta$ ", fontsize=15)

ax[1].set_ylabel(r"Report rate | $\alpha$", fontsize=15)
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels,fontsize=15, loc=[0.85,0.5])
plt.tight_layout()
plt.savefig("figs/inference_results.png")
plt.show()

obs_all = samples['obs_posterior']

confirmed_post_all   = np.mean(obs_all[0,:,:,:],-1)
deaths_post_all      = np.mean(obs_all[1,:,:,:],-1)
df_confirmed   = create_df_response(confirmed_post_all, time=num_steps, dates =dates_assimilation)
df_deaths      = create_df_response(deaths_post_all, time=num_steps, dates =dates_assimilation)

fig, ax = plt.subplots(2, 1, figsize=(17.5, 10), sharex=True)

ax[0].plot(df_confirmed.index.values, df_confirmed["median"], color='gray')#, label='Median')
ax[0].fill_between(df_confirmed.index.values, df_confirmed["high_95"], df_confirmed["low_95"], color='cyan', alpha=0.3)#, label='95% CI')
ax[0].fill_between(df_confirmed.index.values, df_confirmed["high_50"], df_confirmed["low_50"], color='blue', alpha=0.3)#, label='50% CI')
ax[0].scatter(obs_df.index.values, obs_df["smoothed_InfectadosNoVacunados"], edgecolors="w", facecolor="darkred")

ax[1].plot(df_deaths.index.values, df_deaths["median"], color='gray', label='Median')
ax[1].fill_between(df_deaths.index.values, df_deaths["high_95"], df_deaths["low_95"], color='cyan', alpha=0.3, label='95% CI')
ax[1].fill_between(df_deaths.index.values, df_deaths["high_50"], df_deaths["low_50"], color='blue', alpha=0.3, label='50% CI')
ax[1].scatter(obs_df.index.values, obs_df["smoothed_MuertesNoVacunados"], edgecolors="w", facecolor="darkred", label='Observations')

for axi in ax.flatten():
    axi.xaxis.set_major_locator(mdates.MonthLocator())
    axi.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
    axi.xaxis.set_minor_locator(mdates.WeekdayLocator())
    axi.tick_params( which='both', axis='both', labelsize=15)
    axi.tick_params( which='both', axis='x', rotation=90, labelsize=15)

    axi.spines['right'].set_visible(False)
    axi.spines['top'].set_visible(False)


ax[0].set_ylabel("Confirmed Cases", fontsize=15)
ax[1].set_ylabel("Deaths", fontsize=15)
ax[1].legend(fontsize=14, loc='best')
plt.savefig("figs/fitting_model.png")
plt.show()

x_post_all = samples['x_posterior']

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
plt.savefig("figs/state_variables.png")
plt.show()