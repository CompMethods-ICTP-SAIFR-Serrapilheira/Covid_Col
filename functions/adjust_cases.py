# ------------------------------------------------------#
# Scientific computing
# ICTP/Serrapilheira 2022
# Final project: COVID19 Colombia
# First version 2022-08-15
# ------------------------------------------------------#

import pandas as pd
import numpy as np

# Smooths cases using a rolling window and gaussian sampling
def prepare_cases(daily_cases, col='num_cases', out_col=None, cutoff=0):
    if not out_col:
        out_col = 'smoothed_'+str(col)

    daily_cases[out_col] = daily_cases[col].rolling(window=7,
        win_type='gaussian',
        min_periods=1,
        center=True).mean(std=10).round()

    idx_start = np.searchsorted(daily_cases[out_col], cutoff)
    daily_cases[out_col] = daily_cases[out_col].iloc[idx_start:]
    return daily_cases
