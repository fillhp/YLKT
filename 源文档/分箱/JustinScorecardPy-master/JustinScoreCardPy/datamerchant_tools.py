# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 12:07:50 2022

@author: Administrator
"""
import numpy as np
import pandas as pd

def weight_ks(df,bad_weight=1,good_weight=0.33):
    """
    df: dataframe
    bad_weight: 坏客权重
    good_weight:
    """
    d2 = df
    #d2 = pd.DataFrame()
    d2['weight_good'] = d2['good']/good_weight
    d2['weight_bad'] = d2['bad']/bad_weight
    d2['weight_total'] = d2['weight_good'] + d2['weight_bad']
    d2['cum_bad'] = d2['weight_bad'].cumsum()
    d2['cum_good'] = d2['weight_good'].cumsum()
    d2['cum_total'] = d2['weight_total'].cumsum()
    d2['weight_badrate'] = d2['weight_bad']/d2['weight_total']
    d2['weight_goodrate'] = d2['weight_good']/d2['weight_total']
    d2['weight_odds'] = d2['weight_good']/d2['weight_bad']
    d2['ks'] = d2['cum_bad']/d2['cum_bad'].max() - d2['cum_good']/d2['cum_good'].max()
    d2['ks_max'] = np.abs(d2['ks']).max()
    return d2
