"""
    单闪125帧特征计算
    Version：12.2
    Updata time：2023-2-23
"""
import numpy as np
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
                 is_round=True, start=20,
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
