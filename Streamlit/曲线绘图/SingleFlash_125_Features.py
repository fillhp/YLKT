"""
    单闪125帧特征计算
    Version：12.0
    Updata time：2023-1-12
"""

from collections import defaultdict

import numpy as np
from scipy.signal import welch
from scipy.stats import linregress


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


# 一次去除异常值
def pure1(row):
    temp = row
    # 前
    temp1 = temp[0:30]
    median1 = np.median(temp1)
    f1 = 2 - (np.std(temp1) / np.mean(temp1)) * 10  # 根据变异系数判断异常值范围
    for i in range(0, 30):
        if abs(temp[i] - median1) >= f1:
            temp[i] = median1

    # 中 3sigma
    temp2 = temp[30:70]
    mean2 = np.mean(temp2)
    std2 = np.std(temp2, ddof=0)
    for i in range(30, 70):
        if (abs(temp[i] - mean2) > 2 * std2):
            temp[i] = temp[i - 1]
    # 后
    temp3 = temp[70:125]
    median3 = np.median(temp3)
    f3 = 2 - (np.std(temp3) / np.mean(temp3)) * 10
    for i in range(70, 125):
        if abs(temp[i] - median3) >= f3:
            temp[i] = median3
    return temp


# 二次去异常值
def pure2(row):
    temp = row

    # 计算每一个点到前后两个点的平均值的距离

    def getexp(temp):
        h_list = []
        for b in range(1, len(temp) - 1):
            h = abs(temp[b] - ((temp[b - 1] + temp[b + 1]) / 2))
            h_list.append(h)

        h_median = np.median(h_list)
        h_list.insert(0, h_median)  # 在第一个和最后一个位置增加中位数，统一下标
        h_list.append(h_median)

        # 使用箱型图对距离去除异常值
        exp = []
        higher = np.quantile(h_list, 0.75, interpolation='higher')  # 上四分位
        lower = np.quantile(h_list, 0.25, interpolation='lower')  # 下四分位
        iqr = higher - lower  # 四分位距
        low, up = lower - 1.5 * iqr, higher + 1.5 * iqr
        for i in range(len(h_list)):
            if h_list[i] > up or h_list[i] < low:
                exp.append(i)

        for i in exp:
            # i不能等于前一个或后一个数
            if i != 0 or i != len(h_list):  # i不能为第一个或最后一个数
                if temp[i] != temp[i - 1]:
                    temp[i] = temp[i - 1]
                else:
                    temp[i] = (temp[i - 1] + temp[i + 1]) / 2
        return (exp, temp)

    for i in range(10):
        x = getexp(temp)
        if len(x[0]) == 0:
            temp = x[1]
            break
    return temp


# 平滑
def smooth(data, smooth_fraction=0.04, iterations=1):
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

    data = np.array(data)

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

            XtX = (A.transpose(1, 0, 2, 3)[None, ...] * A[:, None, ...]).sum(2)
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

    return list(smooth.T[0])


# 计算特征
class Features():
    def __init__(self, row, is_pure1=True, is_pure2=True, is_smooth=True, is_round=True, start=20, end=101):
        # -----------------------数据处理-----------------------
        if is_pure1:
            row = pure1(row)  # 一次去异常
        if is_pure2:
            row = pure2(row)  # 二次去异常
        if is_smooth:
            row = smooth(row)  # 平滑
        if is_round:
            row = [int(i * 10000) / 10000.0 for i in row]  # 取小数点后四位 不进行四舍五入

        # -----------------------基础特征-----------------------
        # 最小值与下标
        temp = row[30:80]
        minvalue = min(temp)
        minindex = temp.index(minvalue)
        minindex = minindex + 30

        # 最大值与下标
        x1, y1 = 0, np.median(row[:25])  # 利用0：25帧的中位数代替初始值
        x2, y2 = minindex, minvalue
        l = count_l(x1, y1, x2, y2)
        dl_list = []
        for b in range(20, minindex):
            x0, y0 = b, row[b]  # 动点
            dl = count_dl(x0, y0, l)
            dl_list.append(dl)

        maxindex = dl_list.index(max(dl_list))
        maxindex = maxindex + 20
        maxvalue = row[maxindex]
        # 返回值与下标
        x1, y1 = minindex, minvalue
        x2, y2 = 124, np.median(row[80:])  # 利用80：124帧的中位数代替末值
        l = count_l(x1, y1, x2, y2)
        dl_list = []
        for b in range(minindex, 80):
            x0, y0 = b, row[b]
            dl = count_dl(x0, y0, l)
            dl_list.append(dl)

        retindex = dl_list.index(max(dl_list))
        retindex = retindex + minindex
        retvalue = row[retindex]

        self.maxindex = maxindex
        self.maxvalue = maxvalue
        self.minindex = minindex
        self.minvalue = minvalue
        self.retindex = retindex
        self.retvalue = retvalue

        self.row = row
        self.start = start
        self.end = end

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
        return np.sum(np.array(self.row[self.start:self.end]) - self.minvalue)

    # 下降面积
    def get_S_down(self):
        return np.sum(np.array(self.row[self.maxindex:self.minindex + 1]) - self.minvalue)

    # 回归面积
    def get_S_ret(self):
        return np.sum(np.array(self.row[self.minindex:self.retindex + 1]) - self.minvalue)

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
            cosine = np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q))  # 余弦相似度

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
        x = np.array(self.row)
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
        x = np.array(self.row)
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
        x = np.array(self.row)
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
        x = np.array(self.row)
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

        x = np.array(self.row)
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
        x = np.array(self.row)
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
        x = np.array(self.row)
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
                    np.sum(np.array_split(x, num_segments)[segment_focus] ** 2.0)
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


# 规范化数据 （最小值，最大值，下四分位，上四分位）
norm_dict = {"maxindex": [25, 34, 29, 31], "maxvalue": [16.4499, 43.4781, 28.3217, 34.7498],
             "minindex": [39, 57, 44, 49], "minvalue": [9.8564, 36.8564, 19.7282, 25.55], "retindex": [59, 79, 70, 76],
             "retvalue": [14.1281, 43.0, 26.0086, 31.9999],
             "range1": [1.2716999999999992, 14.504000000000001, 6.643500000000003, 9.965699999999998],
             "range2": [1.3999000000000006, 11.6282, 4.849899999999998, 7.121700000000001], "F1": [8, 29, 14, 19],
             "F2": [15, 37, 23, 28],
             "V1": [0.16111111111111112, 0.8229714285714286, 0.42042666666666667, 0.5595142857142857],
             "V2": [0.04916764705882346, 0.4845083333333333, 0.18541249999999998, 0.2860043478260869],
             "V_sum": [0.23075796703296703, 1.2070247619047618, 0.6213889473684211, 0.8335568783068781],
             "S": [97.5615, 683.8321999999998, 333.39599999999996, 455.59819999999996],
             "S_down": [8.950499999999998, 144.67800000000003, 43.7236, 77.5417],
             "S_ret": [14.291499999999996, 165.772, 57.52209999999998, 93.66889999999998],
             "S_low": [32.22349999999999, 258.22489999999993, 109.26589999999996, 163.44789999999992],
             "S_sum": [130.2456662, 933.9921608999998, 442.8085007, 624.7174911],
             "maxang": [2.4115115517182852, 2.9780955880338027, 2.6139869331436056, 2.7373237501763654],
             "minang": [2.0191232918401294, 2.8928010771029657, 2.364667824518967, 2.544975421668391],
             "retang": [2.703542063399037, 3.1037951161433917, 2.887824775906841, 2.9825339918413305],
             "sumang": [5.664188673056097, 7.025875282203273, 6.217328344912953, 6.5059347298818295],
             "var": [0.22935431999999992, 21.462307111662856, 4.493135903633592, 10.223687937146776],
             "std": [0.47890951128579595, 4.77655400193108, 2.119701843098126, 3.1974502243423237],
             "cv": [0.016086479584755915, 0.1884145756475429, 0.07631562553906052, 0.11694174716441817],
             "dmq": [7.924001995159114, 540.9856325533854, 134.4843914529517, 270.42127935921565],
             "dnp": [7.226447378513639e-05, 0.11186272640399553, 0.01557139405109292, 0.0476693202754105],
             "curvature": [67.83036677889145, 203.2619370496727, 91.37201262601641, 129.03094508222844],
             "lumpiness": [0.0004361763179674479, 4.679814778470367, 0.19943793699925924, 1.703670928435074],
             "level_shift_size": [0.06108500000000561, 0.7039049999999953, 0.33531999999999584, 0.5000000000000036],
             "stability": [0.20371569616212096, 17.810797974654953, 3.61631045735497, 8.463182786082063],
             "std1st_der": [0.06473819616393386, 0.4240177038915238, 0.2096649101491233, 0.28873799810180856],
             "mac": [0.052612096774193415, 0.2689395161290323, 0.12997822580645224, 0.18400161290322567],
             "asoc": [6.523899999999983, 33.3485, 16.11730000000008, 22.816199999999984],
             "fft_imag_3": [-10.449332080691475, 106.48094245413593, 31.744023261788936, 59.77914569352065],
             "fft_imag_6": [-22.027311568166656, 19.990409351173888, -5.230738870707297, 3.5248771697329735],
             "fft_angle_3": [66.30620695634163, 179.30483564512414, 114.2587815425192, 145.01309003810783],
             "fft_angle_2": [-150.38246598247883, -61.21323460891936, -115.44925118764195, -96.60804254697952],
             "fft_angle_6": [-179.93840793149772, 179.99248900704197, -162.6991813641646, 154.10830648019746],
             "fft_angle_39": [-178.64855517803673, 179.06548252241552, -37.520980507609615, 123.0787397168268],
             "fft_real_2": [-139.55624743098224, 34.38916814542434, -55.04924821695017, -9.785247159681376],
             "fft_real_5": [-18.336788996105835, 26.833816828193196, -0.5086659555895715, 9.375271977559862],
             "fft_abs_1": [28.304156908295663, 328.6233582114942, 132.63258284932888, 213.34790508348698],
             "fft_abs_22": [0.06867344203989358, 4.666248454119139, 0.9225884870368424, 2.19385350354458],
             "alt_stderr_min_50": [0.0, 3.492969128597237, 1.1772460663911064, 2.0766134482212464],
             "alt_stderr_mean_10": [0.021909791932318387, 0.3463810660692381, 0.1484338827122108, 0.23207506655113527],
             "alt_stderr_mean_5": [0.00853856196371326, 0.1284713093017975, 0.05615824344752844, 0.0870197727975392],
             "alt_stderr_max_5": [0.007680698081626171, 0.1219238440351631, 0.051755191772617136, 0.08169217172157686],
             "alt_stderr_min_5": [0.00958497780252601, 0.1351209964067994, 0.06040512140367653, 0.09239818842090261],
             "alt_stderr_max_10": [0.01906161272055444, 0.30599316863156417, 0.12836155805334068, 0.20312630613697982],
             "alt_rvalue_min_50": [0.7495899755805285, 1.0, 0.8660254037844386, 0.9276138590174807],
             "swd_coeff_2": [7.220922958757079, 1220.7907970764124, 215.01827258489135, 551.1115601148803],
             "erbc_num10_focus2": [0.09000005091926196, 0.13573372637092201, 0.10425237154307074, 0.11484243246443733],
             "erbc_num10_focus4": [0.04958358055987486, 0.10424239264859612, 0.07297250342840653, 0.08938360565514686]}


# 对特征值去异常
def pure3(fea_dict, norm_dict):
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
def goone(fea_dict, norm_dict):
    fea_dict = fea_dict
    for fea in fea_dict.keys():
        value = fea_dict[fea]
        if value > norm_dict[fea][1]:
            fea_dict[fea] = 1
        elif value < norm_dict[fea][0]:
            fea_dict[fea] = 0
        else:
            fea_dict[fea] = (value - norm_dict[fea][0]) / (norm_dict[fea][1] - norm_dict[fea][0])
    return fea_dict


# 计算所有特征
def all_features(row, is_pure1=True, is_pure2=True, is_smooth=True, is_round=True, start=20, end=101, is_goone=True,
                 is_pure3=True):
    fea = Features(row, is_pure1, is_pure2, is_smooth, is_round, start, end)
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
        {"coeff": 6, "attr": "angle"},
        {"coeff": 39, "attr": "angle"},
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
        fea_dict = pure3(fea_dict, norm_dict)
    if is_goone:
        fea_dict = goone(fea_dict, norm_dict)
    return fea_dict
