import pandas as pd
import numpy as np
import math
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import streamlit as st
#---------------------------差值评分---------------------------
@st.cache
def gap(n_fea,p_fea,fea_list):
    gap_dict={}
    for fea in fea_list:
        n_col=list(n_fea[fea])
        p_col=list(p_fea[fea])

        gap_list=[]
        gap_up=0
        gap_down=0
        for a in range(len(n_col)):
            gaps=[]
            for b in range(len(p_col)):
                gap=p_col[b]-n_col[a]
                gaps.append(gap)
                if gap>0:
                    gap_up=gap_up+1
                elif gap<0:
                    gap_down=gap_down+1
            gap_list.append(gaps)

        gap_bili=0
        if (gap_up+gap_down)!=0:
            if gap_up/(gap_up+gap_down)>0.5:
                gap_bili=gap_up/(gap_up+gap_down)
            else:
                gap_bili=gap_down/(gap_up+gap_down)

        gap_dict[fea]=round(gap_bili*100,2)

    gap_dict = dict(sorted(gap_dict.items(), key=lambda x: x[1],reverse=True))
    return gap_dict


#------------------------随机森林------------------------------
@st.cache
def rf(df,fea_list):
    df=df[fea_list+['target']]
    x = df.drop(['target'],axis=1)
    y = df['target']
    rfmodel = RandomForestClassifier(random_state=0)
    rfmodel = rfmodel.fit(x,y)

    keys=list(x.columns)
    values=rfmodel.feature_importances_
    rf_dict={}
    for i in range(len(keys)):
        rf_dict[keys[i]]=values[i]

    rf_dict = dict(sorted(rf_dict.items(), key=lambda x: x[1],reverse=True))
    return rf_dict


#------------------------xgboost筛选变量------------------------------
@st.cache
def xgboost(df,fea_list):
    df=df[fea_list+['target']]
    x = df.drop(['target'],axis=1)
    y = df['target']
    xgmodel = XGBClassifier(random_state=0)
    xgmodel = xgmodel.fit(x,y,eval_metric='auc')

    keys=list(x.columns)
    values=xgmodel.feature_importances_
    xgboost_dict={}
    for i in range(len(keys)):
        xgboost_dict[keys[i]]=float(values[i])

    xgboost_dict = dict(sorted(xgboost_dict.items(), key=lambda x: x[1],reverse=True))
    return xgboost_dict

#---------------------------IV值---------------------------
@st.cache
def iv(data, fea_list):
    iv_dict={}
    for fea in fea_list:
        value_list = set(list(np.unique(data[fea])))
        iv = 0
        data_bad = pd.Series(data[data['target']==1][fea].values, index=data[data['target']==1].index)
        data_good = pd.Series(data[data['target']==0][fea].values, index=data[data['target']==0].index)
        len_bad = len(data_bad)
        len_good = len(data_good)
        for value in value_list:
            # 判断是否某类是否为0，避免出现无穷小值和无穷大值
            if sum(data_bad == value) == 0:
                bad_rate = 1 / len_bad
            else:
                bad_rate = sum(data_bad == value) / len_bad
            if sum(data_good == value) == 0:
                good_rate = 1 / len_good
            else:
                good_rate = sum(data_good == value) / len_good
            iv += (good_rate - bad_rate) * math.log(good_rate / bad_rate,2)
            iv_dict[fea]=iv
    iv_dict = dict(sorted(iv_dict.items(), key=lambda x: x[1],reverse=True))
    return iv_dict

#------------------------综合排名------------------------------
@st.cache
def syn(gap_dict,rf_dict,xgboost_dict,iv_dict):
    gap_keys=list(gap_dict.keys())
    rf_keys=list(rf_dict.keys())
    xgboost_keys=list(xgboost_dict.keys())
    iv_keys=list(iv_dict.keys())
    syn_dict={}
    loser_list=[]
    for fea in gap_keys:
        if gap_dict[fea]<55:
            loser_list.append(fea)
        else:
            rank=gap_keys.index(fea)+rf_keys.index(fea)+xgboost_keys.index(fea)+iv_keys.index(fea)
            syn_dict[fea]=rank
    syn_dict = dict(sorted(syn_dict.items(), key=lambda x: x[1],reverse=False))
    syn_list=list(syn_dict.keys())
    return syn_list,loser_list,syn_dict

#------------------------推荐------------------------------
@st.cache
def recommend(gap_dict,rf_dict,xgboost_dict,iv_dict,syn_list,loser_list,option,threshold):

    def del_loser(ddict,loser_list):
        llist=list(ddict.keys())
        for loser in loser_list:
            if loser in llist:
                llist.remove(loser)
        return llist

    recommend_list=[]
    if option == '考虑综合排名':
        try:
            threshold=int(threshold*len(syn_list)/100)
            recommend_list=syn_list[:threshold]
        except:
            recommend_list=['推荐失败！']
    elif option == '考虑差值评分':
        try:
            gap_list=del_loser(gap_dict,loser_list)
            threshold=int(threshold*len(gap_list)/100)
            recommend_list=gap_list[:threshold]
        except:
            recommend_list=['推荐失败！']
    elif option == '考虑随机森林评分':
        try:
            rf_list=del_loser(rf_dict,loser_list)
            threshold=int(threshold*len(rf_list)/100)
            recommend_list=rf_list[:threshold]
        except:
            recommend_list=['推荐失败！']
    elif option == '考虑xgboost评分':
        try:
            xgboost_list=del_loser(xgboost_dict,loser_list)
            threshold=int(threshold*len(xgboost_list)/100)
            recommend_list=xgboost_list[:threshold]
        except:
            recommend_list=['推荐失败！']
    elif option == '考虑IV评分':
        try:
            iv_list=del_loser(iv_dict,loser_list)
            threshold=int(threshold*len(iv_list)/100)
            recommend_list=iv_list[:threshold]
        except:
            recommend_list=['推荐失败！']
    return recommend_list