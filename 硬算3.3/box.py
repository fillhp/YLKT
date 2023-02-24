import pandas as pd
import numpy as np


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
                axis=0)/(np.ones(freq_array.shape) * freq_array.sum(axis=0).T)
            tmp_abs = abs(tmp.T[0] - tmp.T[1])
            ks_v = tmp_abs.max()
            BestSplit_Point = bincut[tmp_abs.tolist().index(ks_v)]
            BestSplit_Position = tmp_abs.tolist().index(ks_v)/max(len(bincut) - 1, 1)

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
    best_bincut.append(data[var].min()-1)
    # 排序切分点
    best_bincut.sort()

    return best_bincut


def divide(n_fea, p_fea, fea_list):
    config = {}
    np_fea = pd.concat([n_fea, p_fea], axis=0)
    for fea in fea_list:
        config[fea]={}
        n_col = list(n_fea[fea])
        p_col = list(p_fea[fea])

        bins = get_bestks_bincut(np_fea, fea, 'target')

        n_counts = pd.cut(n_col, bins).value_counts()
        p_counts = pd.cut(p_col, bins).value_counts()
        np_counts = p_counts/(n_counts+p_counts)
        np_counts.fillna(0, inplace=True)
        rate = np_counts.tolist()

        config[fea]["bins"] = bins
        config[fea]["rate"] = rate
    return config


def jugde(data, config):
    fea_list = list(config.keys())

    result=[]
    for a in range(len(data)):
        row = data.iloc[a]
        rate = 0
        for fea in fea_list:
            value = row[fea]
            bins = config[fea]["bins"]
            rates = config[fea]["rate"]
            if value < min(bins):  # 比范围最小还小
                rate = rate+rates[0]
            elif value > max(bins):  # 比范围最大还大
                rate = rate+rates[-1]
            else:
                for b in range(len(bins)-1):
                    if value > bins[:-1][b] and value <= bins[1:][b]:
                        rate = rate+rates[b]
        if rate > len(fea_list)*0.5:
            result.append(1)
        else:
            result.append(0)
    return result
