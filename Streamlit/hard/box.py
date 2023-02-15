import pandas as pd
import numpy as np
from sklearn.cluster import KMeans #引入KMeans
import streamlit as st
import math
from scipy.stats import chi2
#---------------------------------固定分箱-------------------------------------
#k均值分箱
@st.cache
def kmeans(np_col,ng):
    k_np_col=[[i] for i in np_col]
    kmodel = KMeans(init="k-means++",n_clusters=ng-1)
    kmodel.fit(k_np_col)
    k_bins = pd.DataFrame(kmodel.cluster_centers_).sort_values(0).values
    bins=[bin[0] for bin in k_bins]
    return bins

#等距分箱
@st.cache
def distance(np_col,ng):
    bins = np.linspace(min(np_col),max(np_col),ng+1)
    return bins

#等频分箱
@st.cache
def frequency(np_col,ng):
    percentiles = np.linspace(0,100,ng+1)
    bins=np.percentile(np_col, percentiles)
    return bins


#----------------------------------------------最优分箱--------------------------------------------￥
#-------------------计算IV---------------------
@st.cache
def iv_count(data, var, target):
    ''' 计算iv值
    Args:
        data: DataFrame，拟操作的数据集
        var: String，拟计算IV值的变量名称
        target: String，Y列名称
    Returns:
        IV值， float
    '''
    value_list = set(list(np.unique(data[var])))
    iv = 0
    data_bad = pd.Series(data[data[target]==1][var].values, index=data[data[target]==1].index)
    data_good = pd.Series(data[data[target]==0][var].values, index=data[data[target]==0].index)
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
        # print(value,iv)
    return iv


#-----------------基于CART算法的最优分箱-------------------
@st.cache
def get_var_median(data, var):
    var_value_list = list(np.unique(data[var]))
    var_median_list = []
    for i in range(len(var_value_list)-1):
        var_median = (var_value_list[i] + var_value_list[i+1]) / 2
        var_median_list.append(var_median)
    return var_median_list

@st.cache
def calculate_gini(y):

    # 将数组转化为列表
    y = y.tolist()
    probs = [y.count(i)/len(y) for i in np.unique(y)]
    gini = sum([p*(1-p) for p in probs])
    return gini

@st.cache
def get_cart_split_point(data, var, target, min_sample):
    # 初始化
    Gini = calculate_gini(data[target].values)
    Best_Gini = 0.0
    BestSplit_Point = -99999
    BestSplit_Position = 0.0
    median_list = get_var_median(data, var) # 获取当前数据集指定元素的所有中位数列表

    for i in range(len(median_list)):
        left = data[data[var] < median_list[i]]
        right = data[data[var] > median_list[i]]

        # 如果切分后的数据量少于指定阈值，跳出本次分箱计算
        if len(left) < min_sample or len(right) < min_sample:
            continue

        Left_Gini = calculate_gini(left[target].values)
        Right_Gini = calculate_gini(right[target].values)
        Left_Ratio = len(left) / len(data)
        Right_Ratio = len(right) / len(data)

        Temp_Gini = Gini - (Left_Gini * Left_Ratio + Right_Gini * Right_Ratio)
        if Temp_Gini > Best_Gini:
            Best_Gini = Temp_Gini
            BestSplit_Point = median_list[i]
            # 获取切分点的位置，最左边为0，最右边为1
            if len(median_list) > 1:
                BestSplit_Position = i / (len(median_list) - 1)
            else:
                BestSplit_Position = i / len(len(median_list))
        else:
            continue
    Gini = Gini - Best_Gini
    # print("最优切分点：", BestSplit_Point)
    return BestSplit_Point, BestSplit_Position

@st.cache
def get_cart_bincut(data, var, target, leaf_stop_percent=0.05):
    min_sample = len(data) * leaf_stop_percent
    best_bincut = []

    def cutting_data(data, var, target, min_sample, best_bincut):
        split_point, position = get_cart_split_point(data, var, target, min_sample)

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
    best_bincut.append(data[var].min()-1)
    # 排序切分点
    best_bincut.sort()

    return best_bincut


 #------------------ 基于卡方检验的最优分箱---------------
@st.cache
def calculate_chi(freq_array):
    """ 计算卡方值
    Args:
        freq_array: Array，待计算卡方值的二维数组，频数统计结果
    Returns:
        卡方值，float
    """
    # 检查是否为二维数组
    assert(freq_array.ndim==2)

    # 计算每列的频数之和
    col_nums = freq_array.sum(axis=0)
    # 计算每行的频数之和
    row_nums = freq_array.sum(axis=1)
    # 计算总频数
    nums = freq_array.sum()
    # 计算期望频数
    E_nums = np.ones(freq_array.shape) * col_nums / nums
    E_nums = (E_nums.T * row_nums).T
    # 计算卡方值
    tmp_v = (freq_array - E_nums)**2 / E_nums
    # 如果期望频数为0，则计算结果记为0
    tmp_v[E_nums==0] = 0
    chi_v = tmp_v.sum()
    return chi_v

@st.cache
def get_chimerge_bincut(data, var, target, max_group=None, chi_threshold=None):
    freq_df = pd.crosstab(index=data[var], columns=data[target])
    # 转化为二维数组
    freq_array = freq_df.values

    # 初始化箱体，每个元素单独一组
    best_bincut = freq_df.index.values

    # 初始化阈值 chi_threshold，如果没有指定 chi_threshold，则默认选择target数量-1，置信度95%来设置阈值
    if max_group is None:
        if chi_threshold is None:
            chi_threshold = chi2.isf(0.05, df = freq_array.shape[-1])

    # 开始迭代
    while True:
        min_chi = None
        min_idx = None
        for i in range(len(freq_array) - 1):
            # 两两计算相邻两组的卡方值，得到最小卡方值的两组
            v = calculate_chi(freq_array[i: i+2])
            if min_chi is None or min_chi > v:
                min_chi = v
                min_idx = i
        
        # 是否继续迭代条件判断
        # 条件1：当前箱体数仍大于 最大分箱数量阈值
        # 条件2：当前最小卡方值仍小于制定卡方阈值
        if (max_group is not None and max_group < len(freq_array)) or (chi_threshold is not None and min_chi < chi_threshold):
            tmp = freq_array[min_idx] + freq_array[min_idx+1]
            freq_array[min_idx] = tmp
            freq_array = np.delete(freq_array, min_idx+1, 0)
            best_bincut = np.delete(best_bincut, min_idx+1, 0)
        else:
            break
    
    # 把切分点补上头尾
    best_bincut = best_bincut.tolist()
    best_bincut.append(data[var].min())
    best_bincut.append(data[var].max())
    best_bincut_set = set(best_bincut)
    best_bincut = list(best_bincut_set)
    
    best_bincut.remove(data[var].min())
    best_bincut.append(data[var].min()-1)
    # 排序切分点
    best_bincut.sort()
    
    return best_bincut

#----------------- 基于最优KS的最优分箱---------------------
@st.cache
def get_maxks_split_point(data, var, target, min_sample=0.05):

    if len(data) < min_sample:
        ks_v, BestSplit_Point, BestSplit_Position = 0, -9999, 0.0
    else:
        freq_df = pd.crosstab(index=data[var], columns=data[target])
        freq_array = freq_df.values
        if freq_array.shape[1] == 1: # 如果某一组只有一个枚举值，如0或1，则数组形状会有问题，跳出本次计算
            # tt = np.zeros(freq_array.shape).T
            # freq_array = np.insert(freq_array, 0, values=tt, axis=1)
            ks_v, BestSplit_Point, BestSplit_Position = 0, -99999, 0.0
        else:
            bincut = freq_df.index.values
            tmp = freq_array.cumsum(axis=0)/(np.ones(freq_array.shape) * freq_array.sum(axis=0).T)
            tmp_abs = abs(tmp.T[0] - tmp.T[1])
            ks_v = tmp_abs.max()
            BestSplit_Point = bincut[tmp_abs.tolist().index(ks_v)]
            BestSplit_Position = tmp_abs.tolist().index(ks_v)/max(len(bincut) - 1, 1)
        
    return ks_v, BestSplit_Point, BestSplit_Position

@st.cache
def get_bestks_bincut(data, var, target, leaf_stop_percent=0.05):
    min_sample = len(data) * leaf_stop_percent
    best_bincut = []
    
    def cutting_data(data, var, target, min_sample, best_bincut):
        ks, split_point, position = get_maxks_split_point(data, var, target, min_sample)
        
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
    best_bincut.append(data[var].min()-1)
    # 排序切分点
    best_bincut.sort()
    
    return best_bincut

#开分！
@st.cache
def divide(n_fea,p_fea,fea_list,ng,number,option):
    config={"way":"固定分箱|"+option+"|"+str(ng)+"组|"+str(number)+"项","threshold":0.0,"rate":{},"is_fit":True,"fea":{}}

    for fea in fea_list:
        config["fea"][fea]={}
        n_col=list(n_fea[fea])
        p_col=list(p_fea[fea])
        np_col=n_col+p_col

        if option=="k-means":
            bins = kmeans(np_col,ng)
        elif option=="等距分箱":
            bins = distance(np_col,ng)
        elif option=="等频分箱":
            bins = frequency(np_col,ng)

        n_counts = pd.cut(n_col,bins).value_counts()
        p_counts = pd.cut(p_col,bins).value_counts()
        np_counts=p_counts/(n_counts+p_counts)
        np_counts.fillna(0, inplace=True)
        rate=np_counts.tolist()

        x=[i for i in range(len(bins)-1)]
        z = np.polyfit(x,rate, number)
        eq = np.poly1d(z)
        fit_rate=list(eq(x))

        config["fea"][fea]["bins"]=bins
        config["fea"][fea]["z"]=list(z)
        config["fea"][fea]["rate"]=rate
        config["fea"][fea]["fit_rate"]=fit_rate

    return config


@st.cache
def best_divide(n_fea,p_fea,fea_list,number,option):
    config={"way":"自动分箱|"+option+"|"+str(number)+"项","threshold":0.0,"rate":{},"is_fit":True,"fea":{}}
    np_fea=pd.concat([n_fea,p_fea],axis=0)
    for fea in fea_list:
        config["fea"][fea]={}
        n_col=list(n_fea[fea])
        p_col=list(p_fea[fea])

        if option=="最优分箱":
            bins1 = get_cart_bincut(np_fea, fea, 'target')
            bins2 = get_chimerge_bincut(np_fea, fea, 'target')
            bins3 = get_bestks_bincut(np_fea, fea, 'target')
            np_fea[fea+'_bins1'] = pd.cut(np_fea[fea], bins=bins1)
            np_fea[fea+'_bins2'] = pd.cut(np_fea[fea], bins=bins2)
            np_fea[fea+'_bins3'] = pd.cut(np_fea[fea], bins=bins3)
            binss=[bins1,bins2,bins3]
            IV1=iv_count(np_fea, fea+'_bins1', 'target')
            IV2=iv_count(np_fea, fea+'_bins2', 'target')
            IV3=iv_count(np_fea, fea+'_bins3', 'target')
            IVs=[IV1,IV2,IV3]
            best=IVs.index(max(IVs))
            bins=binss[best]
        elif option=="CART算法":
            bins=get_cart_bincut(np_fea, fea, 'target')
        elif option=="卡方检验":
            bins=get_chimerge_bincut(np_fea, fea, 'target')
        elif option=="最优KS":
            bins=get_bestks_bincut(np_fea, fea, 'target')

        n_counts = pd.cut(n_col,bins).value_counts()
        p_counts = pd.cut(p_col,bins).value_counts()
        np_counts=p_counts/(n_counts+p_counts)
        np_counts.fillna(0, inplace=True)
        rate=np_counts.tolist()

        x=[i for i in range(len(bins)-1)]
        z = np.polyfit(x,rate, number)
        eq = np.poly1d(z)
        fit_rate=list(eq(x))

        config["fea"][fea]["bins"]=bins
        config["fea"][fea]["z"]=list(z)
        config["fea"][fea]["rate"]=rate
        config["fea"][fea]["fit_rate"]=fit_rate
    return config

@st.cache
def jugde(data,config,threshold,is_fit):
    p=0
    fea_list=list(config["fea"].keys())
    for a in range(len(data)):
        row=data.iloc[a]
        rate=0
        for fea in fea_list:
            value=row[fea]
            bins=config["fea"][fea]["bins"]
            if value<min(bins): #比范围最小还小
                rate=0.01
            elif value>max(bins): #比范围最大还大
                rate=1
            else:
                for b in range(len(bins)-1):
                    if value>bins[:-1][b] and value<=bins[1:][b]:
                        if is_fit:
                            rate=rate+config["fea"][fea]["fit_rate"][b]
                        else:
                            rate=rate+config["fea"][fea]["rate"][b]
        if rate>len(fea_list)*threshold:
            p=p+1
    return p