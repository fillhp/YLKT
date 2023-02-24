"""
    单闪125帧特征计算
    Version：12.2
    Updata time：2023-2-23
"""

from collections import defaultdict

import numpy as np
from scipy.signal import welch
from scipy.stats import linregress

# 规范化数据 （最小值，最大值，下四分位，上四分位）
norm_dict = {"S_down": [1.75884688, 145.7076136, 44.47784295, 84.44711327], "maxvalue": [15.68746624, 43.77175318, 28.63333155, 35.19319712], "range1": [0.704396538, 13.38848062, 6.524962755, 9.899946008], "range2": [1.499999855, 10.0844182, 4.559405879, 6.449972499], "dmq": [0.966208655, 113.6418168, 26.49958079, 57.6571724], "V_sum": [0.230553622, 1.095989514, 0.579121659, 0.763652529], "S_ret": [10.41089006, 163.9491651, 58.48011095, 95.1405328], "curvature": [46.16894861, 209.1907801, 91.46261479, 130.7594887], "alt_stderr_mean_5": [0.007407655, 0.128095097, 0.05557646, 0.088283085], "alt_rvalue_min_50": [0.770345813, 0.999993149, 0.866025397, 0.92912082], "V1": [0.140879308, 0.784699698, 0.394959787, 0.524999047], "V2": [0.044228662, 0.378301919, 0.174998063, 0.243634999], "stability": [0.218932633, 18.23994501, 3.541933792, 8.778755507], "std": [0.472115, 4.573664734, 2.089421895, 3.163033888], "level_shift_size": [0.049999984, 0.669358745, 0.329507595, 0.499074123], "var": [0.222892574, 20.9184091, 4.365683857, 10.00478338], "dnp": [9.07e-05, 0.090347043, 0.013828361, 0.039608947], "fft_abs_1": [27.94323126, 369.5061535, 132.1111367, 222.8684578], "F1": [5, 30, 15, 21], "asoc": [6.003475309, 31.26056507, 15.65608855, 21.48124095], "F2": [14, 36, 24, 29], "erbc_num10_focus4": [0.050875065, 0.10462811, 0.073506712, 0.089506778], "erbc_num10_focus2": [0.083989992, 0.134836962, 0.104035127, 0.114320649], "maxindex": [27, 32, 29, 30], "minang": [2.114588685, 2.895986157, 2.422778204, 2.583449603], "fft_imag_3": [-15.1146756, 90.09423731, 29.4324307, 54.10446375], "fft_imag_6": [-20.94816742, 16.56990138, -
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       6.122955241, 1.795979436], "fft_abs_22": [0.024530824, 4.595098737, 0.847835719, 2.105043083], "lumpiness": [0.000296, 3.715053583, 0.187283259, 1.388950639], "minindex": [39, 59, 44, 50], "alt_stderr_max_10": [0.016473564, 0.3243065, 0.127961686, 0.210363685], "retvalue": [14.69997311, 41.94997314, 25.94997261, 31.94997274], "sumang": [5.83697202, 7.049152643, 6.322849686, 6.568553552], "mac": [0.048415123, 0.252101331, 0.126258779, 0.173235814], "S": [99.36423379, 655.2255798, 326.5490704, 443.382116], "retang": [2.836644711, 3.099354661, 2.945699603, 3.003694933], "S_low": [23.81942921, 269.3866935, 112.5924837, 175.4617556], "retindex": [55, 79, 70, 78], "S_sum": [91.86478646, 882.3214023, 444.0494976, 628.9551678], "alt_stderr_min_5": [0.008128787, 0.13448872, 0.059740273, 0.093270072], "minvalue": [9.979504246, 38.82511933, 19.99999995, 26.29234326], "alt_stderr_mean_10": [0.019374464, 0.41244123, 0.146893949, 0.238080811], "swd_coeff_2": [5.576804057, 1087.062657, 209.4785026, 524.6132202], "std1st_der": [0.066572909, 0.389677405, 0.202214119, 0.271911971], "cv": [0.015614734, 0.190752531, 0.074311718, 0.113206033], "fft_angle_2": [-138.8847905, -56.36004788, -116.2539994, -96.1793463], "maxang": [2.45578527, 2.97768312, 2.646933254, 2.757328125], "fft_real_2": [-108.0039609, 33.53963029, -55.34960605, -8.998279754], "fft_real_5": [-15.43324935, 22.0158669, -1.202244949, 6.676385617], "alt_stderr_min_50": [0.003686757, 3.462218706, 1.165415834, 2.10188244], "alt_stderr_max_5": [0.00668768, 0.144735484, 0.051707343, 0.083237744], "fft_angle_3": [74.1762466, 179.7406766, 119.3570562, 146.4043123]}


# 求直线
def count_l(x1, y1, x2, y2):
    l = {
        'A': 1,
        'B': (x1 - x2) / (y2 - y1),
        'C': ((x1 - x2) * y1 / (y2 - y1)) - x1
    }
    return l


# 两点距离
def count_dd(x1, y1, x2, y2):
    d = pow(pow(x1 - x2, 2) + pow(y1 - y2, 2), 0.5)
    return d


# 动点到直线距离
def count_dl(x0, y0, l):
    A = l['A']
    B = l['B']
    C = l['C']
    dl = abs(A * x0 + B * y0 + C) / pow(pow(A, 2) + pow(B, 2), 0.5)
    return dl


# 计算夹角
def count_ang(x1_, y1_, x_, y_, x2_, y2_):
    """
        计算三个点之间的夹角的角度
        公式：
            point1(x,y)
            point2(x1,y1)
            point3(x2,y2)
            cos(A) = (x2-x1)*(x3-x1) + (y2-y1)*(y3-y1) / sqrt([(x2-x1)*(x2-x1) + (y2-y1)*(y2-y1)]*[(x3-x1)*(x3-x1)+(y3-y1)*(y3-y1)]);
        input:
            (x1_,y1_),(x2_,y2_):夹角的两个端点
            (x_,y_):被求夹角的点
        return:
            arc_cos_a:夹角角度
    """
    try:
        cos_a = ((x1_ - x_) * (x2_ - x_) + (y1_ - y_) * (y2_ - y_)) / pow(
            (pow(x1_ - x_, 2) + pow(y1_ - y_, 2)) * (pow(x2_ - x_, 2) + pow(y2_ - y_, 2)), 0.5)
        arc_cos_a = np.arccos(cos_a)
    except:
        arc_cos_a = 0
    return arc_cos_a


# 计算特征
class Features():
    def __init__(self, row, is_pure1=True, is_pure2=True, max_pure_num=10, exception_threshold=0.5, is_smooth=True,
                 is_round=False, start=20,
                 end=101):
        self.row = np.array(row)
        self.ini_row = self.row.copy()
        self.maxindex, self.maxvalue, self.minindex, self.minvalue, self.retindex, self.retvalue = 0, 0, 0, 0, 0, 0

        # 数据包含空值或数据长度不足125帧
        if len(self.row) != 125:
            self.exception = True
        else:
            try:
                for pure_num in range(max_pure_num):  # max_pure_num:最多循环去异常次数
                    keep = (self.pure1() if is_pure1 else True,
                            self.pure2() if is_pure2 else True)

                    if False not in keep:  # 如果已经没有异常的帧
                        break  # 提前结束

                if is_smooth:
                    self.smooth()  # 平滑
                if is_round:
                    # 取小数点后四位 不进行四舍五入
                    # self.row = np.array([int(i*10000)/10000.0 for i in self.row])
                    self.row = np.around(self.row, 4)

                self.get_base()  # 获取基础特征

                if self.maxvalue < self.minvalue or self.retvalue < self.minvalue or self.maxindex > self.minindex or self.minindex > self.retindex:
                    self.exception = True
                else:
                    self.judge_exception(exception_threshold)
                    self.start = start
                    self.end = end
            except:
                self.exception = True

    def judge_exception(self, exception_threshold):
        # exception_threshold [0,1] 值越大，判断异常的标准越严格
        # 当exception_threshold=1时只有会导致计算报错才判断 exception = True

        is_exception = False
        if exception_threshold != 1:
            # 异常修复面积
            def get_fix_s(ini_row, row):
                return np.sum(abs(ini_row - row) / np.mean(ini_row))

            # 异常修复次数
            def get_fix_num(ini_row, row):
                fix_num_list = []
                for i in range(len(ini_row)):
                    if (abs(ini_row[i] - row[i]) + row[i]) == 0:
                        fix_num_list.append(-1)
                    elif 1 - (row[i] / (abs(ini_row[i] - row[i]) + row[i])) > 0.05:
                        if ini_row[i] > row[i]:
                            fix_num_list.append(1)
                        else:
                            fix_num_list.append(-1)
                    else:
                        fix_num_list.append(0)

                zero = []
                for i in range(1, len(fix_num_list)):
                    if i != 0:
                        if fix_num_list[i] == fix_num_list[i - 1]:
                            zero.append(i)

                for i in zero:
                    fix_num_list[i] = 0.1  # 连贯的突起约视为一次

                return sum(np.abs(fix_num_list))

            # 异常修复比率的中位数和均值
            def get_fix_extent(ini_row, row, way):
                fixup = 1 - (row / (np.abs(ini_row - row) + row))
                for i in range(len(fixup)):
                    if np.isnan(fixup[i]):
                        fixup[i] = 0.5
                if way == "median":
                    return np.median(fixup)
                else:
                    return np.mean(fixup)

            # 异常修复的标准差
            def get_fix_var(ini_row, row):
                return np.sum(np.power(ini_row - row, 2)) / 125

            ini_row1 = self.ini_row[:self.maxindex]
            ini_row2 = self.ini_row[self.maxindex:self.retindex]
            ini_row3 = self.ini_row[self.retindex:]
            row1 = self.row[:self.maxindex]
            row2 = self.row[self.maxindex:self.retindex]
            row3 = self.row[self.retindex:]

            fix_dict = {
                "fix_s1": get_fix_s(ini_row1, row1),
                "fix_s2": get_fix_s(ini_row2, row2),
                "fix_s3": get_fix_s(ini_row3, row3),

                "fix_num1": get_fix_num(ini_row1, row1),
                "fix_num2": get_fix_num(ini_row2, row2),
                "fix_num3": get_fix_num(ini_row3, row3),

                "fix_extent_median1": get_fix_extent(ini_row1, row1, "median"),
                "fix_extent_median2": get_fix_extent(ini_row2, row2, "median"),
                "fix_extent_median3": get_fix_extent(ini_row3, row3, "median"),

                "fix_extent_mean1": get_fix_extent(ini_row1, row1, "mean"),
                "fix_extent_mean2": get_fix_extent(ini_row2, row2, "mean"),
                "fix_extent_mean3": get_fix_extent(ini_row3, row3, "mean"),

                "fix_var1": get_fix_var(ini_row1, row1),
                "fix_var2": get_fix_var(ini_row2, row2),
                "fix_var3": get_fix_var(ini_row3, row3),
            }

            exception_threshold += 0.5
            threshold_dict = {
                'fix_s1': 0.04293427529626193,
                'fix_s2': 0.17816964809384006,
                'fix_s3': 0.1393172685489167,

                'fix_num1': 6.199999999999994,
                'fix_num2': 7.399999999999999,
                'fix_num3': 9.999999999999993,

                'fix_extent_median1': 0.015873015873015928,
                'fix_extent_median2': 0.01005490196078429,
                'fix_extent_median3': 0.008964346883298502,

                'fix_extent_mean1': 0.04843168802260982,
                'fix_extent_mean2': 0.02830448409587059,
                'fix_extent_mean3': 0.037478812762854495,

                'fix_var1': 9.419213279600001,
                'fix_var2': 5.451495381795556,
                'fix_var3': 15.89516533392}

            judge_dict = {}
            for key in threshold_dict.keys():
                if fix_dict[key] >= threshold_dict[key] * exception_threshold:
                    judge_dict[key] = 1
                else:
                    judge_dict[key] = 0

            # 中
            if judge_dict['fix_num2'] and (judge_dict['fix_extent_median2'] or judge_dict['fix_var2']):
                is_exception = True

            else:
                # 若两边修复后较稳定，变异系数较小，则判为正常数据
                if np.std(row1) / np.mean(row1) < 0.05 and np.std(row3) / np.mean(row3) < 0.05:
                    is_exception = False

                # 前
                elif sum([judge_dict['fix_extent_mean1'], judge_dict['fix_num1'], judge_dict['fix_s1'],
                          judge_dict['fix_var1']]) >= 2:
                    is_exception = True

                # 后
                elif sum([judge_dict['fix_extent_mean3'], judge_dict['fix_num3'], judge_dict['fix_s3'],
                          judge_dict['fix_var3']]) >= 2:
                    is_exception = True

        self.exception = is_exception

    # -----------------------一次去异常值-----------------------
    # 宏观的从一整段数据去异常
    def pure1(self):
        pure_ini_row = self.row.copy()

        # 前
        temp1 = self.row[0:30]
        median1 = np.median(temp1)
        f1 = 2 - (np.std(temp1) / np.mean(temp1)) * 10  # 根据变异系数判断异常值范围
        for i in range(0, 30):
            if abs(self.row[i] - median1) >= f1:
                self.row[i] = median1

        # 中 3sigma
        temp2 = self.row[30:70]
        mean2 = np.mean(temp2)
        std2 = np.std(temp2)
        for i in range(30, 70):
            if abs(self.row[i] - mean2) > 2 * std2:
                self.row[i] = (self.row[i - 1] + self.row[i + 1]) / 2

        # 后1 3sigma
        temp3 = self.row[70:100]
        mean3 = np.mean(temp3)
        std3 = np.std(temp3)
        for i in range(70, 100):
            if abs(self.row[i] - mean3) > 2 * std3:
                self.row[i] = (self.row[i - 1] + self.row[i + 1]) / 2

        # 后2
        temp4 = self.row[100:125]
        median4 = np.median(temp4)
        f4 = 2 - (np.std(temp4) / np.mean(temp4)) * 10  # 根据变异系数判断异常值范围
        for i in range(100, 125):
            if abs(self.row[i] - median4) >= f4:
                self.row[i] = median4

        if (pure_ini_row == self.row).all():  # 初始数据和去异常后的数据每一帧都相等
            return True  # 不用继续去异常
        else:
            return False

    # -----------------------二次去异常值-----------------------
    # 微观的从邻近数据去异常
    def pure2(self):
        pure_ini_row = self.row.copy()
        # 计算每一个点到前后两个点的平均值的距离
        h_array = np.abs(self.row[1:-1] - ((self.row[:-2] + self.row[2:]) / 2))
        # 使用箱型图对距离去除异常值
        higher = np.quantile(
            h_array, 0.75, interpolation='higher')  # 上四分位
        lower = np.quantile(
            h_array, 0.25, interpolation='lower')  # 下四分位

        iqr = higher - lower  # 四分位距

        for i in range(1, 124):
            if h_array[i - 1] > higher + 3 * iqr:  # 波动较大 用附近10帧的中位数代替
                if i + 6 > 125:
                    self.row[i] = np.median(self.row[114:])
                elif i - 5 < 0:
                    self.row[i] = np.median(self.row[:11])
                else:
                    self.row[i] = np.median(self.row[i - 5:i + 6])

            elif h_array[i - 1] > higher + iqr:  # 波动较小 用左右2帧的平均数代替
                self.row[i] = (self.row[i - 1] + self.row[i + 1]) / 2

        if (pure_ini_row == self.row).all():  # 初始数据和去异常后的数据每一帧都相等
            return True  # 不用继续去异常
        else:
            return False

    # -----------------------平滑-----------------------
    def smooth(self, smooth_fraction=0.04, iterations=1):
        """
        取自：tsmoothie
        LowessSmoother uses LOWESS (locally-weighted scatterplot smoothing)
        to smooth the timeseries. This smoothing technique is a non-parametric
        regression method that essentially fit a unique linear regression
        for every data point by including nearby data points to estimate
        the slope and intercept. The presented method is robust because it
        performs residual-based reweightings simply specifing the number of
        iterations to operate.

        The LowessSmoother automatically vectorizes, in an efficient way,
        the desired smoothing operation on all the series passed.

        Parameters
        ----------
        smooth_fraction : float
            Between 0 and 1. The smoothing span. A larger value of smooth_fraction
            will result in a smoother curve.

        iterations : int
            Between 1 and 6. The number of residual-based reweightings to perform.

        """

        data = self.row
        if smooth_fraction >= 1 or smooth_fraction <= 0:
            raise ValueError("smooth_fraction must be in the range (0,1)")

        if iterations <= 0 or iterations > 6:
            raise ValueError("iterations must be in the range (0,6]")

        if data.ndim == 1:
            data = data[:, None]

        timesteps, n_timeseries = data.shape

        X = np.arange(timesteps) / (timesteps - 1)

        # Create basis for LOWESS.
        xx = np.arange(timesteps)
        r = int(np.ceil(smooth_fraction * timesteps))
        r = min(r, timesteps - 1)
        xx = xx[:, None] - xx[None, :]
        h = np.sort(np.abs(xx), axis=1)[:, r]
        X_base = np.abs(xx / h).clip(0.0, 1.0)
        w_init = np.power(1 - np.power(X_base, 3), 3)

        delta = np.ones_like(data)
        batches = [np.arange(0, n_timeseries)]
        smooth = np.empty_like(data)

        for iteration in range(iterations):
            for B in batches:
                w = delta[:, None, B] * w_init[..., None]
                # (timesteps, timesteps, n_series)
                wy = w * data[:, None, B]
                # (timesteps, timesteps, n_series)
                wyx = wy * X[:, None, None]
                # (timesteps, timesteps, n_series)
                wx = w * X[:, None, None]
                # (timesteps, timesteps, n_series)
                wxx = wx * X[:, None, None]
                # (timesteps, timesteps, n_series)

                b = np.array([wy.sum(axis=0), wyx.sum(axis=0)]).T
                # (n_series, timesteps, 2)
                A = np.array([[w.sum(axis=0), wx.sum(axis=0)],
                              [wx.sum(axis=0), wxx.sum(axis=0)]])
                # (2, 2, timesteps, n_series)

                XtX = (A.transpose(1, 0, 2, 3)[
                    None, ...] * A[:, None, ...]).sum(2)
                # (2, 2, timesteps, n_series)
                XtX = np.linalg.pinv(XtX.transpose(3, 2, 0, 1))
                # (n_series, timesteps, 2, 2)
                XtXXt = (XtX[..., None] * A.transpose(3, 2, 1, 0)
                         [..., None, :]).sum(2)
                # (n_series, timesteps, 2, 2)
                betas = np.squeeze(XtXXt @ b[..., None], -1)
                # (n_series, timesteps, 2)

                smooth[:, B] = (betas[..., 0] + betas[..., 1] * X).T
                # (timesteps, n_series)

                residuals = data[:, B] - smooth[:, B]
                s = np.median(np.abs(residuals), axis=0).clip(1e-5)
                delta[:, B] = (residuals / (6.0 * s)).clip(-1, 1)
                delta[:, B] = np.square(1 - np.square(delta[:, B]))

        self.row = smooth.T[0]

    # -----------------------基础特征-----------------------
    def get_base(self):
        # 最小值与下标
        temp = list(self.row[30:80])
        minvalue = min(temp)
        minindex = temp.index(minvalue)
        minindex = minindex + 30

        # 最大值与下标
        x1, y1 = 0, np.median(self.row[:25])  # 利用0：25帧的中位数代替初始值
        x2, y2 = minindex, minvalue
        l = count_l(x1, y1, x2, y2)
        dl_list = []
        for b in range(20, minindex):
            x0, y0 = b, self.row[b]  # 动点
            dl = count_dl(x0, y0, l)
            dl_list.append(dl)

        maxindex = dl_list.index(max(dl_list))
        maxindex = maxindex + 20
        maxvalue = self.row[maxindex]
        # 返回值与下标
        x1, y1 = minindex, minvalue
        x2, y2 = 124, np.median(self.row[80:])  # 利用80：124帧的中位数代替末值
        l = count_l(x1, y1, x2, y2)
        dl_list = []
        for b in range(minindex, 80):
            x0, y0 = b, self.row[b]
            dl = count_dl(x0, y0, l)
            dl_list.append(dl)

        retindex = dl_list.index(max(dl_list))
        retindex = retindex + minindex
        retvalue = self.row[retindex]

        self.maxindex = maxindex
        self.maxvalue = maxvalue
        self.minindex = minindex
        self.minvalue = minvalue
        self.retindex = retindex
        self.retvalue = retvalue

    # -----------------------基础特征衍生-----------------------
    # 最大值-最小值

    def get_range1(self):
        return self.maxvalue - self.minvalue

    # 返回值-最小值
    def get_range2(self):
        return self.retvalue - self.minvalue

    # 最小值坐标-最大值坐标
    def get_F1(self):
        return self.minindex - self.maxindex

    # 返回值-最小值坐标
    def get_F2(self):
        return self.retindex - self.minindex

    # 从最大值下降到最小值所用速度
    def get_V1(self):
        try:
            return self.get_range1() / self.get_F1()
        except:
            return 0

    # 从最小值上升到返回值所用速度
    def get_V2(self):
        try:
            return self.get_range2() / self.get_F2()
        except:
            return 0

    # 下降和上升的速度和
    def get_V_sum(self):
        return self.get_V1() + self.get_V2()

    # -----------------------面积特征-----------------------
    # 从设定的开始帧到结束帧计算

    # 整体面积
    def get_S(self):
        return np.sum(self.row[self.start:self.end] - self.minvalue)

    # 下降面积
    def get_S_down(self):
        return np.sum(self.row[self.maxindex:self.minindex + 1] - self.minvalue)

    # 回归面积
    def get_S_ret(self):
        return np.sum(self.row[self.minindex:self.retindex + 1] - self.minvalue)

    # 低谷面积
    def get_S_low(self):
        return self.get_S_down() + self.get_S_ret()

    # 面积和
    def get_S_sum(self):
        return self.get_S() * 0.826 + self.get_S_down() * 0.776 + self.get_S_ret() * 0.723 + self.get_S_low() * 0.773

    # -----------------------角度特征-----------------------
    # 将最大值，最小值，返回值三点连线，组成一个三角形，求各个角的角度

    # 最大值所在夹角
    def get_maxang(self):
        x0, y0 = 0, self.row[0]  # 起点
        x1, y1 = self.maxindex, self.maxvalue  # 最大点
        x2, y2 = self.minindex, self.minvalue  # 最低点
        return count_ang(x0, y0, x1, y1, x2, y2)

    # 最小值所在夹角
    def get_minang(self):
        x1, y1 = self.maxindex, self.maxvalue  # 最大点
        x2, y2 = self.minindex, self.minvalue  # 最低点
        x3, y3 = self.retindex, self.retvalue  # 回归点
        return count_ang(x1, y1, x2, y2, x3, y3)

    # 返回值所在夹角
    def get_retang(self):
        x2, y2 = self.minindex, self.minvalue  # 最低点
        x3, y3 = self.retindex, self.retvalue  # 回归点
        x4, y4 = len(self.row) - 1, self.row[len(self.row) - 1]  # 终点
        return count_ang(x2, y2, x3, y3, x4, y4)

    # 角度和
    def get_sumang(self):
        return self.get_maxang() * 0.812 + self.get_minang() * 0.819 + self.get_retang() * 0.741

    # -----------------------发散程度-----------------------
    # 方差
    def get_var(self):
        return np.var(self.row[self.start:self.end])

    # 标准差
    def get_std(self):
        return np.std(self.row[self.start:self.end])

    # 变异系数
    def get_cv(self):
        return np.std(self.row[self.start:self.end]) / np.mean(self.row[self.start:self.end])

    # -----------------------D-P 计算特征-----------------------
    """
        反映数据离中趋势的特征
        参考文章：https://www.sohu.com/a/445945937_120381558
    """

    # 每一个动点到前后两个点的连线的距离的方差

    def get_dmq(self):
        temp = self.row[self.start:self.end]
        m_list = []
        for i in range(1, len(temp) - 1):
            x, y = i, temp[i]  # 动点
            x1, y1 = i - 1, temp[i - 1]
            x2, y2 = i + 1, temp[i + 1]
            if y1 == y2:
                m = y - y2
            else:
                nl = count_l(x1, y1, x2, y2)  # 每一个动点的前后两个点的连线
                m = count_dl(x, y, nl)
            m_list.append(m)

        return np.var(m_list)

    # 每一个动点的前后两个点的距离减去这组数据的中位数的差的平方
    def get_dnp(self):
        temp = self.row[self.start:self.end]
        n_list = []
        for i in range(1, len(temp) - 1):
            x1, y1 = i - 1, temp[i - 1]
            x2, y2 = i + 1, temp[i + 1]
            n = count_dd(x1, y1, x2, y2)
            n_list.append(n)

        dnp = np.sum(pow(np.array(n_list) - np.median(n_list), 2))
        return dnp / (self.end - self.start)

    # -----------------------曲率-----------------------
    def get_curvature(self):
        """
        意义：
            曲率是针对曲线上某个点的切线方向角对弧长的转动率，通过微分来定义，表明曲线偏离直线的程度。数学上表明曲线在某一点的弯曲程度的数值。曲率越大，表示曲线的弯曲程度越大。
        公式：
            \frac{1}{2a}\sqrt{(a+b+c)(a+b-c)(a+c-b)(b+c-a)}
        """

        # 列表组合成二维数据
        pts = list(zip([i for i in range(self.end - self.start)],
                       self.row[self.start:self.end]))  # 从设定的开始帧到结束帧计算

        # 计算弦长
        start = np.array(pts[0])
        end = np.array(pts[len(pts) - 1])
        l_arc = np.sqrt(np.sum(np.power(end - start, 2)))

        # 计算弧上的点到直线的最大距离
        a = l_arc
        b = np.sqrt(np.sum(np.power(pts - start, 2), axis=1))
        c = np.sqrt(np.sum(np.power(pts - end, 2), axis=1))
        dist = np.sqrt((a + b + c) * (a + b - c) *
                       (a + c - b) * (b + c - a)) / (2 * a)
        h = dist.max()

        return ((a * a) / 4 + h * h) / (2 * h)

    # -----------------------相似性 特征-----------------------
    def get_similarity(self, side_data):
        similarity_list = []
        for i in range(len(side_data)):
            side_row = np.array(side_data.iloc[i])
            p, q = np.array(self.row), side_row
            euclidean = np.sqrt(np.sum(np.square(p - q)))  # 欧几里得距离
            # similarity=np.linalg.norm(np.array(self.row) - side_row, ord=1) #曼哈顿距离
            cosine = np.dot(p, q) / (np.linalg.norm(p)
                                     * np.linalg.norm(q))  # 余弦相似度

            similarity_list.append(abs(cosine) / euclidean)
        similarity_median = np.median(similarity_list)
        return similarity_median

    # -----------------------kats 特征-----------------------
    """
        引用时间序列特征计算包：Kats
        x: 一行数据的数组
        window_size: 滑动窗口长度，默认为20
    """

    # lumpiness

    def get_lumpiness(self, window_size=20):
        """
        Calculating the lumpiness of time series.
        Lumpiness is defined as the variance of the chunk-wise variances.

        """
        x = self.row
        v = [np.var(x_w)
             for x_w in np.array_split(x, len(x) // window_size + 1)]
        return np.var(v)

    # level_shift_size
    def get_level_shift_size(self, window_size=20):
        """
        Calculate level shift features.

        level_shift_size: Size of the maximum mean value difference,
        between two consecutive sliding windows

        """
        x = self.row
        sliding_idx = (np.arange(len(x))[None, :] + np.arange(window_size)[:, None])[
            :, : len(x) - window_size + 1
        ]
        means = np.mean(x[sliding_idx], axis=0)
        mean_diff = np.abs(means[:-1] - means[1:])
        return mean_diff[np.argmax(mean_diff)]

    # stability
    def get_stability(self, window_size=20):
        """
        Calculate the stability of time series.
        Stability is defined as the variance of chunk-wise means.

        """
        x = self.row
        v = [np.mean(x_w)
             for x_w in np.array_split(x, len(x) // window_size + 1)]
        return np.var(v)

    # std1st_der
    def get_std1st_der(self):
        """
        Calculate the standard deviation of the first derivative of the time series.

        Reference: https://cran.r-project.org/web/packages/tsfeatures/vignettes/tsfeatures.html
        """
        return np.std(np.gradient(np.array(self.row)))

    # ------------------------tsfresh 特征--------------------------
    """
        引用时间序列特征计算包：tsfresh
        源码文档：https://tsfresh.readthedocs.io/en/latest/_modules/tsfresh/feature_extraction/feature_calculators.html#fft_coefficient
    """

    # 傅里叶变换

    def get_fft(self, param):
        """
        Calculates the fourier coefficients of the one-dimensional discrete Fourier Transform for real input by fast
        fourier transformation algorithm

        .. math::
            A_k =  \\sum_{m=0}^{n-1} a_m \\exp \\left \\{ -2 \\pi i \\frac{m k}{n} \\right \\}, \\qquad k = 0,
            \\ldots , n-1.

        The resulting coefficients will be complex, this feature calculator can return the real part (attr=="real"),
        the imaginary part (attr=="imag), the absolute value (attr=""abs) and the angle in degrees (attr=="angle).

        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :param param: contains dictionaries {"coeff": x, "attr": s} with x int and x >= 0, s str and in ["real", "imag",
            "abs", "angle"]
        :type param: list
        :return: the different feature values
        :return type: dict
        """
        x = self.row
        assert (
            min([config["coeff"] for config in param]) >= 0
        ), "Coefficients must be positive or zero."
        assert {config["attr"] for config in param} <= {
            "imag",
            "real",
            "abs",
            "angle",
        }, 'Attribute must be "real", "imag", "angle" or "abs"'

        fft = np.fft.rfft(x)

        def complex_agg(x, agg):
            if agg == "real":
                return x.real
            elif agg == "imag":
                return x.imag
            elif agg == "abs":
                return np.abs(x)
            elif agg == "angle":
                return np.angle(x, deg=True)

        res = [
            complex_agg(fft[config["coeff"]], config["attr"])
            if config["coeff"] < len(fft)
            else np.NaN
            for config in param
        ]
        index = [
            # 'attr_"{}"__coeff_{}'.format(config["attr"], config["coeff"])
            'fft_' + config["attr"] + '_' + str(config["coeff"])
            for config in param
        ]
        return dict(zip(index, res))

    # 时间序列值的线性最小二乘回归(agg_linear_trend)

    def get_alt(self, param):
        """
        Calculates a linear least-squares regression for values of the time series that were aggregated over chunks versus
        the sequence from 0 up to the number of chunks minus one.

        This feature assumes the signal to be uniformly sampled. It will not use the time stamps to fit the model.

        The parameters attr controls which of the characteristics are returned. Possible extracted attributes are "pvalue",
        "rvalue", "intercept", "slope", "stderr", see the documentation of linregress for more information.

        The chunksize is regulated by "chunk_len". It specifies how many time series values are in each chunk.

        Further, the aggregation function is controlled by "f_agg", which can use "max", "min" or , "mean", "median"

        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :param param: contains dictionaries {"attr": x, "chunk_len": l, "f_agg": f} with x, f an string and l an int
        :type param: list
        :return: the different feature values
        :return type: dict
        """

        def _aggregate_on_chunks(x, f_agg, chunk_len):
            return [
                getattr(x[i * chunk_len: (i + 1) * chunk_len], f_agg)()
                for i in range(int(np.ceil(len(x) / chunk_len)))
            ]

        x = self.row
        calculated_agg = defaultdict(dict)
        res_data = []
        res_index = []

        for parameter_combination in param:

            chunk_len = parameter_combination["chunk_len"]
            f_agg = parameter_combination["f_agg"]

            if f_agg not in calculated_agg or chunk_len not in calculated_agg[f_agg]:
                if chunk_len >= len(x):
                    calculated_agg[f_agg][chunk_len] = np.NaN
                else:
                    aggregate_result = _aggregate_on_chunks(
                        x, f_agg, chunk_len)
                    lin_reg_result = linregress(
                        range(len(aggregate_result)), aggregate_result
                    )
                    calculated_agg[f_agg][chunk_len] = lin_reg_result

            attr = parameter_combination["attr"]

            if chunk_len >= len(x):
                res_data.append(np.NaN)
            else:
                res_data.append(
                    getattr(calculated_agg[f_agg][chunk_len], attr))

            res_index.append(
                # 'attr_"{}"__chunk_len_{}__f_agg_"{}"'.format(attr, chunk_len, f_agg)
                'alt_' + attr + '_' + f_agg + "_" + str(chunk_len)
            )

        return dict(zip(res_index, res_data))

    # 交叉功率谱密度(spkt_welch_density)
    def get_swd(self, param):
        """
        This feature calculator estimates the cross power spectral density of the time series x at different frequencies.
        To do so, the time series is first shifted from the time domain to the frequency domain.

        The feature calculators returns the power spectrum of the different frequencies.

        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :param param: contains dictionaries {"coeff": x} with x int
        :type param: list
        :return: the different feature values
        :return type: dict
        """
        x = self.row
        freq, pxx = welch(x, nperseg=min(len(x), 256))
        coeff = [config["coeff"] for config in param]
        indices = ["swd_coeff_{}".format(i) for i in coeff]

        if len(pxx) <= np.max(
                coeff
        ):  # There are fewer data points in the time series than requested coefficients

            # filter coefficients that are not contained in pxx
            reduced_coeff = [
                coefficient for coefficient in coeff if len(pxx) > coefficient]
            not_calculated_coefficients = [
                coefficient for coefficient in coeff if coefficient not in reduced_coeff
            ]

            # Fill up the rest of the requested coefficients with np.NaNs
            return dict(zip(
                indices,
                list(pxx[reduced_coeff]) + [np.NaN] *
                len(not_calculated_coefficients),
            ))
        else:
            return dict(zip(indices, pxx[coeff]))

    # energy_ratio_by_chunks

    def get_erbc(self, param):
        """
        Calculates the sum of squares of chunk i out of N chunks expressed as a ratio with the sum of squares over the whole
        series.

        Takes as input parameters the number num_segments of segments to divide the series into and segment_focus
        which is the segment number (starting at zero) to return a feature on.

        If the length of the time series is not a multiple of the number of segments, the remaining data points are
        distributed on the bins starting from the first. For example, if your time series consists of 8 entries, the
        first two bins will contain 3 and the last two values, e.g. `[ 0.,  1.,  2.], [ 3.,  4.,  5.]` and `[ 6.,  7.]`.

        Note that the answer for `num_segments = 1` is a trivial "1" but we handle this scenario
        in case somebody calls it. Sum of the ratios should be 1.0.

        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :param param: contains dictionaries {"num_segments": N, "segment_focus": i} with N, i both ints
        :return: the feature values
        :return type: dict
        """
        x = self.row
        res_data = []
        res_index = []
        full_series_energy = np.sum(x ** 2)

        for parameter_combination in param:
            num_segments = parameter_combination["num_segments"]
            segment_focus = parameter_combination["segment_focus"]
            assert segment_focus < num_segments
            assert num_segments > 0

            if full_series_energy == 0:
                res_data.append(np.NaN)
            else:
                res_data.append(
                    np.sum(np.array_split(x, num_segments)
                           [segment_focus] ** 2.0)
                    / full_series_energy
                )

            res_index.append(
                # "num_segments_{}__segment_focus_{}".format(num_segments, segment_focus)
                "erbc_num{}_focus{}".format(num_segments, segment_focus)
            )

        return dict(zip(res_index, res_data))

    # 绝对差的平均值 (mean_abs_change)

    def get_mac(self):
        """
        Average over first differences.

        Returns the mean over the absolute differences between subsequent time series values which is

        .. math::

            \\frac{1}{n-1} \\sum_{i=1,\\ldots, n-1} | x_{i+1} - x_{i}|

        """
        return np.mean(np.abs(np.diff(self.row)))

    # 连续更改绝对值的总和 (absolute_sum_of_changes)
    def get_asoc(self):
        """
        Returns the sum over the absolute value of consecutive changes in the series x

        .. math::

            \\sum_{i=1, \\ldots, n-1} \\mid x_{i+1}- x_i \\mid

        """
        return np.sum(np.abs(np.diff(self.row)))


# 对特征值去异常
def pure3(fea_dict, norm_dict=norm_dict):
    fea_dict = fea_dict
    for fea in fea_dict.keys():
        iqr = norm_dict[fea][3] - norm_dict[fea][2]
        low, up = norm_dict[fea][2] - 2 * iqr, norm_dict[fea][3] + 2 * iqr
        low_, up_ = norm_dict[fea][2] - 1 * iqr, norm_dict[fea][3] + 1 * iqr
        value = fea_dict[fea]
        if value > up:
            fea_dict[fea] = up_
        elif value < low:
            fea_dict[fea] = low_
    return fea_dict


# 归一化
def goone(fea_dict, norm_dict=norm_dict):
    fea_dict = fea_dict
    for fea in fea_dict.keys():
        value = fea_dict[fea]
        if value > norm_dict[fea][1]:
            fea_dict[fea] = 1
        elif value < norm_dict[fea][0]:
            fea_dict[fea] = 0
        else:
            fea_dict[fea] = (value - norm_dict[fea][0]) / \
                            (norm_dict[fea][1] - norm_dict[fea][0])
    return fea_dict


# 计算所有特征
def all_features(row, is_pure1=True, is_pure2=True, max_pure_num=10, exception_threshold=0.5, is_smooth=True, is_round=False, start=20, end=101, is_goone=True, is_pure3=True):
    fea = Features(row, is_pure1, is_pure2, max_pure_num,
                   exception_threshold, is_smooth, is_round, start, end)
    if fea.exception:
        fea_dict = "数据异常"
    else:
        fea_dict = {
            "maxindex": fea.maxindex,
            "maxvalue": fea.maxvalue,
            "minindex": fea.minindex,
            "minvalue": fea.minvalue,
            "retindex": fea.retindex,
            "retvalue": fea.retvalue,

            "range1": fea.get_range1(),
            "range2": fea.get_range2(),
            "F1": fea.get_F1(),
            "F2": fea.get_F2(),
            "V1": fea.get_V1(),
            "V2": fea.get_V2(),
            "V_sum": fea.get_V_sum(),

            "S": fea.get_S(),
            "S_down": fea.get_S_down(),
            "S_ret": fea.get_S_ret(),
            "S_low": fea.get_S_low(),
            "S_sum": fea.get_S_sum(),

            "maxang": fea.get_maxang(),
            "minang": fea.get_minang(),
            "retang": fea.get_retang(),
            "sumang": fea.get_sumang(),

            "var": fea.get_var(),
            "std": fea.get_std(),
            "cv": fea.get_cv(),

            "dmq": fea.get_dmq(),
            "dnp": fea.get_dnp(),

            "curvature": fea.get_curvature(),

            "lumpiness": fea.get_lumpiness(),
            "level_shift_size": fea.get_level_shift_size(),
            "stability": fea.get_stability(),
            "std1st_der": fea.get_std1st_der(),

            "mac": fea.get_mac(),
            "asoc": fea.get_asoc(),
        }

        fft_param = [
            {"coeff": 3, "attr": "imag"},
            {"coeff": 6, "attr": "imag"},
            {"coeff": 3, "attr": "angle"},
            {"coeff": 2, "attr": "angle"},
            {"coeff": 2, "attr": "real"},
            {"coeff": 5, "attr": "real"},
            {"coeff": 1, "attr": "abs"},
            {"coeff": 22, "attr": "abs"},
        ]
        fea_dict.update(fea.get_fft(fft_param))

        alt_param = [
            {"attr": "stderr", "chunk_len": 50, "f_agg": "min"},
            {"attr": "stderr", "chunk_len": 10, "f_agg": "mean"},
            {"attr": "stderr", "chunk_len": 5, "f_agg": "mean"},
            {"attr": "stderr", "chunk_len": 5, "f_agg": "max"},
            {"attr": "stderr", "chunk_len": 5, "f_agg": "min"},
            {"attr": "stderr", "chunk_len": 10, "f_agg": "max"},

            {"attr": "rvalue", "chunk_len": 50, "f_agg": "min"},
        ]
        fea_dict.update(fea.get_alt(alt_param))

        swd_param = [
            {"coeff": 2}
        ]
        fea_dict.update(fea.get_swd(swd_param))

        erbc_param = [
            {"num_segments": 10, "segment_focus": 2},
            {"num_segments": 10, "segment_focus": 4}
        ]
        fea_dict.update(fea.get_erbc(erbc_param))

        if is_pure3:
            fea_dict = pure3(fea_dict)
        if is_goone:
            fea_dict = goone(fea_dict)
    return fea_dict


# 计算硬算3.3所需特征
def hard_features(row):
    fea = Features(row)
    if fea.exception:
        fea_dict = "数据异常"
    else:
        fea_dict = {
            "minindex": fea.minindex,
            "minvalue": fea.minvalue,
            "retindex": fea.retindex,

            "S_down": fea.get_S_down(),

            "cv": fea.get_cv(),

            "lumpiness": fea.get_lumpiness(),
            "asoc": fea.get_asoc(),
        }

        fft_param = [
            {"coeff": 3, "attr": "imag"},
            {"coeff": 3, "attr": "angle"},
            {"coeff": 2, "attr": "angle"},
            {"coeff": 2, "attr": "real"},
        ]
        fea_dict.update(fea.get_fft(fft_param))

        alt_param = [
            {"attr": "stderr", "chunk_len": 50, "f_agg": "min"},
            {"attr": "stderr", "chunk_len": 5, "f_agg": "max"},
            {"attr": "rvalue", "chunk_len": 50, "f_agg": "min"},
        ]
        fea_dict.update(fea.get_alt(alt_param))

        swd_param = [
            {"coeff": 2}
        ]
        fea_dict.update(fea.get_swd(swd_param))

        erbc_param = [
            {"num_segments": 10, "segment_focus": 2},
            {"num_segments": 10, "segment_focus": 4}
        ]
        fea_dict.update(fea.get_erbc(erbc_param))

    fea_dict = pure3(fea_dict)
    fea_dict = goone(fea_dict)
    return fea_dict


# 计算极度随机树1.1所需特征
def etc_features(row):
    fea = Features(row)
    if fea.exception:
        fea_dict = "数据异常"
    else:
        fea_dict = {
            "minindex": fea.minindex,
            "minvalue": fea.minvalue,
            "retindex": fea.retindex,

            "F2": fea.get_F2(),
            "V2": fea.get_V2(),

            "S_down": fea.get_S_down(),

            "lumpiness": fea.get_lumpiness(),

            "mac": fea.get_mac(),
        }

        fft_param = [
            {"coeff": 3, "attr": "imag"},
            {"coeff": 6, "attr": "imag"},
            {"coeff": 3, "attr": "angle"},
            {"coeff": 2, "attr": "angle"},
            {"coeff": 2, "attr": "real"},
            {"coeff": 5, "attr": "real"},
        ]
        fea_dict.update(fea.get_fft(fft_param))

        alt_param = [
            {"attr": "stderr", "chunk_len": 50, "f_agg": "min"},
            {"attr": "rvalue", "chunk_len": 50, "f_agg": "min"},
        ]
        fea_dict.update(fea.get_alt(alt_param))

        erbc_param = [
            {"num_segments": 10, "segment_focus": 2},
            {"num_segments": 10, "segment_focus": 4}
        ]
        fea_dict.update(fea.get_erbc(erbc_param))

        fea_dict = pure3(fea_dict)
        fea_dict = goone(fea_dict)
        return fea_dict
