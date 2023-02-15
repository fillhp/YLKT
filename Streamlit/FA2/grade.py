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

#------------------------硬算------------------------------
@st.cache
def hard(np_fea,fea_list):
    n_fea = np_fea[np_fea['target']== 0]
    p_fea = np_fea[np_fea['target']== 1]
    # ----------------- 基于最优KS的最优分箱---------------------
    def get_maxks_split_point(data, var, target, min_sample=0.05):
        if len(data) < min_sample:
            ks_v, BestSplit_Point, BestSplit_Position = 0, -9999, 0.0
        else:
            freq_df = pd.crosstab(index=data[var], columns=data[target])
            freq_array = freq_df.values
            if freq_array.shape[1] == 1:  # 如果某一组只有一个枚举值，如0或1，则数组形状会有问题，跳出本次计算
                # tt = np.zeros(freq_array.shape).T
                # freq_array = np.insert(freq_array, 0, values=tt, axis=1)
                ks_v, BestSplit_Point, BestSplit_Position = 0, -99999, 0.0
            else:
                bincut = freq_df.index.values
                tmp = freq_array.cumsum(
                    axis=0) / (np.ones(freq_array.shape) * freq_array.sum(axis=0).T)
                tmp_abs = abs(tmp.T[0] - tmp.T[1])
                ks_v = tmp_abs.max()
                BestSplit_Point = bincut[tmp_abs.tolist().index(ks_v)]
                BestSplit_Position = tmp_abs.tolist().index(ks_v) / max(len(bincut) - 1, 1)

        return ks_v, BestSplit_Point, BestSplit_Position

    def get_bestks_bincut(data, var, target, leaf_stop_percent=0.05):
        min_sample = len(data) * leaf_stop_percent
        best_bincut = []

        def cutting_data(data, var, target, min_sample, best_bincut):
            ks, split_point, position = get_maxks_split_point(
                data, var, target, min_sample)
            if split_point != -99999:
                best_bincut.append(split_point)

            # 根据最优切分点切分数据集，并对切分后的数据集递归计算切分点，直到满足停止条件
            # print("本次分箱的值域范围为{0} ~ {1}".format(data[var].min(), data[var].max()))
            left = data[data[var] < split_point]
            right = data[data[var] > split_point]

            # 当切分后的数据集仍大于最小数据样本要求，则继续切分
            if len(left) >= min_sample and position not in [0.0, 1.0]:
                cutting_data(left, var, target, min_sample, best_bincut)
            else:
                pass
            if len(right) >= min_sample and position not in [0.0, 1.0]:
                cutting_data(right, var, target, min_sample, best_bincut)
            else:
                pass
            return best_bincut

        best_bincut = cutting_data(data, var, target, min_sample, best_bincut)

        # 把切分点补上头尾
        best_bincut.append(data[var].min())
        best_bincut.append(data[var].max())
        best_bincut_set = set(best_bincut)
        best_bincut = list(best_bincut_set)

        best_bincut.remove(data[var].min())
        best_bincut.append(data[var].min() - 1)
        # 排序切分点
        best_bincut.sort()

        return best_bincut

    config = {}
    np_fea = pd.concat([n_fea, p_fea], axis=0)
    for fea in fea_list:
        config[fea] = {}
        n_col = list(n_fea[fea])
        p_col = list(p_fea[fea])
        bins = get_bestks_bincut(np_fea, fea, 'target')
        n_counts = pd.cut(n_col, bins).value_counts()
        p_counts = pd.cut(p_col, bins).value_counts()
        np_counts = p_counts / (n_counts + p_counts)
        np_counts.fillna(0, inplace=True)
        rate = np_counts.tolist()

        config[fea]["bins"] = bins
        config[fea]["rate"] = rate

    fea_result = {}
    for a in range(len(np_fea)):
        row = np_fea.iloc[a]
        for fea in fea_list:
            if a == 0:
                fea_result[fea] = []
            value = row[fea]
            bins = config[fea]["bins"]
            rates = config[fea]["rate"]
            if value < min(bins):  # 比范围最小还小
                rate = rates[0]
            elif value > max(bins):  # 比范围最大还大
                rate = rates[-1]
            else:
                for b in range(len(bins) - 1):
                    if value > bins[:-1][b] and value <= bins[1:][b]:
                        rate = rates[b]
            if rate > 0.5:
                fea_result[fea].append(1)
            else:
                fea_result[fea].append(0)
    fea_result['target'] = list(np_fea['target'])
    rank_dict = {}
    fea_result = pd.DataFrame(fea_result)

    for i in range(len(fea_result)):
        row = fea_result.iloc[i]
        tag = row['target']
        for fea in fea_list:
            if i == 0:
                rank_dict[fea] = 0
            if row[fea] == tag:
                rank_dict[fea] += 1

    for fea in fea_list:
        rank_dict[fea] = rank_dict[fea] / len(np_fea)

    hard_dict = dict(sorted(rank_dict.items(), key=lambda x: x[1], reverse=True))
    return hard_dict

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
def recommend(gap_dict,hard_dict,rf_dict,xgboost_dict,syn_list,loser_list,option,threshold):

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
    elif option == '考虑硬算评分':
        try:
            hard_list=del_loser(hard_dict,loser_list)
            threshold=int(threshold*len(hard_list)/100)
            recommend_list=hard_list[:threshold]
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

    return recommend_list