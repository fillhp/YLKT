# 导入库
import pandas as pd
import numpy as np
import streamlit as st

from pyecharts.charts import Grid, Bar, Timeline, Line, Scatter, HeatMap,Boxplot
from pyecharts.commons.utils import JsCode
import pyecharts.options as opts
from pyecharts.globals import ThemeType

from sklearn.decomposition import PCA

# --------------------------------稳定性排名--------------------------------------------
def mark_std_rank(n_fea, p_fea, fea_list):
    std_rank={'n_std':[],'p_std':[],'np_std':[]}
    for fea in fea_list:
        n_std=np.std(n_fea[fea])
        p_std = np.std(p_fea[fea])
        np_std=(n_std+p_std)/2
        std_rank['n_std'].append(n_std)
        std_rank['p_std'].append(p_std)
        std_rank['np_std'].append(np_std)

    std_rank=pd.DataFrame(std_rank,index=fea_list)
    std_rank.sort_values("np_std", inplace=True)

    std_rank_chart = (
        Bar()
        .add_xaxis(list(std_rank.index))
        .add_yaxis("平均", list(std_rank['np_std']), stack="stack1",itemstyle_opts=opts.ItemStyleOpts(color='purple',opacity=0.4))
        .add_yaxis("阴性", list(std_rank['n_std']), stack="stack1",itemstyle_opts=opts.ItemStyleOpts(color='#5DADE2',opacity=0.7))
        .add_yaxis("阳性", list(std_rank['p_std']), stack="stack1",itemstyle_opts=opts.ItemStyleOpts(color='#EC7063',opacity=0.8))
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="特征值稳定性"),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(interval=0,rotate=-45)),
            )

    )
    return std_rank_chart

# --------------------------------箱型图--------------------------------------------
def mark_boxplot(n_fea, p_fea, fea_list,tit1,tit2,tit3):
    n_data=[]
    p_data=[]
    for fea in fea_list:
        n_data.append(list(n_fea[fea]))
        p_data.append(list(p_fea[fea]))

    boxplot = Boxplot()
    boxplot.add_xaxis(fea_list)
    boxplot.add_yaxis(tit1, boxplot.prepare_data(n_data))# ,itemstyle_opts=opts.ItemStyleOpts(color='#fe8c7f', opacity=0.7)
    boxplot.add_yaxis(tit2, boxplot.prepare_data(p_data))

    boxplot.set_global_opts(title_opts=opts.TitleOpts(title=tit3),
                                datazoom_opts=opts.DataZoomOpts(pos_bottom=0),
                                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(interval=0,rotate=-45)),)
    return boxplot



# --------------------------------特征相关性热力图--------------------------------------------
# 获取相关性数据

@st.cache
def get_hadata(np_fea, fea_list):
    cor = np_fea.iloc[:, :].corr()
    hatdata = []
    for a in range(len(fea_list)):
        fea = fea_list[a]
        col = cor[fea]
        for b in range(len(fea_list)):
            fea = fea_list[b]
            y = col[fea]
            hatdata.append([b, a, round(abs(y), 2)])
    return hatdata
# 画图


def mark_hat(np_fea, fea_list):
    hat = (
        HeatMap()
        .add_xaxis(fea_list)
        .set_global_opts(
            title_opts=opts.TitleOpts(title="特征相似度热力图"),
            visualmap_opts=opts.VisualMapOpts(
                min_=0, max_=1, is_calculable=True, orient="horizontal", pos_right="15%", pos_top='0'
            ),
            legend_opts=opts.LegendOpts(selected_mode='single'),
            xaxis_opts=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(rotate=-45)),
        )
    )

    hat.add_yaxis(
        "",
        fea_list,
        get_hadata(np_fea, fea_list),
        label_opts=opts.LabelOpts(is_show=True, position="inside"),
    )
    return hat

# ------------------------------------------时间 散点图----------------------------------------------
def mark_time_sd(n_fea, p_fea, add_fea,add_kind,fea_list):
    # 寻找临界
    @st.cache
    def get_crit(n_fea, p_fea, fea_list):
        crit = {}
        for fea in fea_list:
            n_col = list(n_fea[fea])
            p_col = list(p_fea[fea])
            np_col = n_col+p_col
            s_dict = {'s': [], 'value': [], 'plou': []}
            for i in range(len(np_col)):
                s = np.sum(np.array(n_col) -
                           np_col[i])+np.sum(np.array(p_col)-np_col[i])
                s_dict['s'].append(abs(s))
                s_dict['value'].append(np_col[i])
                if np.sum(np.array(p_col)-np_col[i]) >= 0:
                    s_dict['plou'].append('up')
                else:
                    s_dict['plou'].append('down')
            minindex = s_dict['s'].index(min(s_dict['s']))
            crit[fea] = [s_dict['value'][minindex], s_dict['plou'][minindex]]

        # 寻找阴阳聚集点
        for fea in fea_list:
            n_col = list(n_fea[fea])
            p_col = list(p_fea[fea])
            if crit[fea][1] == "up":
                p_col.sort(reverse=True)  # 倒序
                crit[fea].append(p_col[len(p_col)*97//100])  # 阴性聚集点

                n_col.sort(reverse=False)  # 正序
                crit[fea].append(n_col[len(n_col)*97//100])  # 阳性聚集点

            else:
                p_col.sort(reverse=False)  # 倒序
                crit[fea].append(p_col[len(p_col)*97//100])  # 阴性聚集点

                n_col.sort(reverse=True)  # 正序
                crit[fea].append(n_col[len(n_col)*97//100])  # 阳性聚集点
        return crit

    # 检出率 误判率 精确率
    @st.cache
    def get_rate(n_fea, p_fea, fea_list, crit):
        rate_dict = {}
        for fea in fea_list:
            rate_dict[fea] = [0, 0, 0, 0, 0]  # [检出人数，误判人数，检出率，误判率，精确率]

        for i in range(len(n_fea)):
            row = n_fea.iloc[i]
            for fea in fea_list:
                value = row[fea]
                if crit[fea][1] == 'up':  # 误判+1
                    if value > crit[fea][0]:
                        rate_dict[fea][1] = rate_dict[fea][1]+1
                else:
                    if value < crit[fea][0]:
                        rate_dict[fea][1] = rate_dict[fea][1]+1

        for i in range(len(p_fea)):
            row = p_fea.iloc[i]
            for fea in fea_list:
                value = row[fea]
                if crit[fea][1] == 'up':  # 检出+1
                    if value > crit[fea][0]:
                        rate_dict[fea][0] = rate_dict[fea][0]+1
                else:
                    if value < crit[fea][0]:
                        rate_dict[fea][0] = rate_dict[fea][0]+1

        for fea in fea_list:
            rate_dict[fea] = [
                round(rate_dict[fea][0]*100/len(p_fea), 1),  # 检出率
                round(rate_dict[fea][1]*100/len(n_fea), 1),  # 误判率
                round((rate_dict[fea][0]+len(n_fea)-rate_dict[fea]
                      [1])*100/(len(n_fea)+len(p_fea)), 1)  # 精确率
            ]
        return rate_dict
    if add_kind==0:#阳性被过采样
        crit = get_crit(pd.concat([n_fea, add_fea], axis=0), p_fea, fea_list)
        rate_dict = get_rate(pd.concat([n_fea, add_fea], axis=0), p_fea, fea_list, crit)
        add_tit = "阴性过采样数据"
    elif add_kind==1:#阴性被过采样
        crit = get_crit(n_fea, pd.concat([p_fea, add_fea], axis=0), fea_list)
        rate_dict = get_rate(n_fea, pd.concat([p_fea, add_fea], axis=0), fea_list, crit)
        add_tit="阳性过采样数据"
    else:#未过采样
        crit = get_crit(n_fea, p_fea, fea_list)
        rate_dict = get_rate(n_fea, p_fea, fea_list, crit)
        add_tit="未被过采样"

    # 画图
    # width="1900px", height="980px",
    time_sd = Timeline(init_opts=opts.InitOpts(theme=ThemeType.WALDEN))
    n_name = list(n_fea.iloc[:, 0])
    p_name = list(p_fea.iloc[:, 0])
    if add_kind!=2:
        add_name=list(add_fea.iloc[:,0])
    for fea in fea_list:
        n_col = list(n_fea[fea])
        p_col = list(p_fea[fea])
        if add_kind!=2:
            add_col = list(add_fea[fea])
            slen = max([len(n_col), len(p_col),len(add_col)])
        else:
            slen = max([len(n_col), len(p_col)])
    # --------------------------散点图------------------------------------------------
        sd = (
            Scatter(
                init_opts=opts.InitOpts(theme=ThemeType.WESTEROS)
            )
            .add_xaxis([i for i in range(slen)])

            .add_yaxis(
                series_name="阴性",
                y_axis=[list(z) for z in zip(n_col, n_name)],
                symbol_size=6,
                label_opts=opts.LabelOpts(is_show=False),

            )

            .add_yaxis(
                series_name="阳性",
                y_axis=[list(z) for z in zip(p_col, p_name)],
                symbol_size=6,
                label_opts=opts.LabelOpts(is_show=False),
                itemstyle_opts=opts.ItemStyleOpts(color='#EC7063')
            )



            .set_global_opts(
                title_opts=opts.TitleOpts(title=fea,
                                          pos_left='center',
                                          title_textstyle_opts=opts.TextStyleOpts(
                                              font_size=33, font_weight='bolder', color='#516b91')
                                          ),
                legend_opts=opts.LegendOpts(pos_top='5%',),

                xaxis_opts=opts.AxisOpts(is_show=False),
                yaxis_opts=opts.AxisOpts(is_show=False, is_scale=True),
                tooltip_opts=opts.TooltipOpts(
                    formatter=JsCode(
                        "function (params) {return params.value[2];}"
                    ),
                    axis_pointer_type="cross",
                )

            )
        )
        if add_kind!=2:
            sd.add_yaxis(
                    series_name=add_tit,
                    y_axis=[list(z) for z in zip(add_col, add_name)],
                    symbol_size=6,
                    label_opts=opts.LabelOpts(is_show=False),
                    itemstyle_opts=opts.ItemStyleOpts(color='#F5B041')
                )

    # --------------------------散点图界限------------------------------------------------
        sd.set_series_opts(
            label_opts=opts.LabelOpts(is_show=False),
        )

        crit_line = (
            Line()
            .add_xaxis([i for i in range(slen)])
            .add_yaxis("临界线",
                       [crit[fea][0] for i in range(slen)],
                       is_symbol_show=False,
                       label_opts=opts.LabelOpts(is_show=False),
                       itemstyle_opts=opts.ItemStyleOpts(color='#8E44AD'),
                       linestyle_opts=opts.LineStyleOpts(width=0.6),
                       )
            .add_yaxis("阴性聚集点",
                       [crit[fea][2] for i in range(slen)],
                       is_symbol_show=False,
                       is_selected=False,
                       label_opts=opts.LabelOpts(is_show=False),
                       itemstyle_opts=opts.ItemStyleOpts(color='blue'),
                       linestyle_opts=opts.LineStyleOpts(
                           width=0.6, type_="dashed"),
                       )
            .add_yaxis("阳性聚集点",
                       [crit[fea][3] for i in range(slen)],
                       is_symbol_show=False,
                       is_selected=False,
                       label_opts=opts.LabelOpts(is_show=False),
                       itemstyle_opts=opts.ItemStyleOpts(color='red'),
                       linestyle_opts=opts.LineStyleOpts(
                           width=0.6, type_="dashed"),
                       )
            .set_global_opts(
                yaxis_opts=opts.AxisOpts(is_show=False),
                xaxis_opts=opts.AxisOpts(is_show=False))
        )

        sd = sd.overlap(crit_line)

    # --------------------------检出率 误判率 精确率------------------------------------------------
        # 颜色范围

        normal = {
            "normal": {
                "color": JsCode(
                    """
                function (params) {
                if (params.value >=80 && params.value <= 100)
                return 'rgba(205, 92, 92,0.8)';
                else if (params.value >=60  && params.value < 80)
                return 'rgba(240, 128, 128,0.8)';
                else if (params.value >=40 && params.value < 60)
                return 'rgba(58, 182, 232,0.8)';
                else if (params.value >=20  && params.value < 40)
                return 'rgba(23, 165, 137,0.8)';
                else return 'rgba(163, 228, 215,0.8)';
                }
                """
                ),
                "barBorderRadius": [0, 7, 7, 0],
            }
        }

        bar = (
            Bar(init_opts=opts.InitOpts(theme=ThemeType.WALDEN))
            .add_xaxis(['检出率', '误判率', '精确率'])
            .add_yaxis("", rate_dict[fea], itemstyle_opts=normal)
            .reversal_axis()
            .set_global_opts(xaxis_opts=opts.AxisOpts(is_show=False,))  # 隐藏Y轴
            .set_series_opts(
                label_opts=opts.LabelOpts(
                    font_size=13,
                    position="inside",
                    formatter="{c}%",
                    color="#F8F9F9"
                ),
            )
        )

    # --------------------------合并------------------------------------------------
        grid = (
            Grid()
            .add(sd, grid_opts=opts.GridOpts())
            .add(bar, grid_opts=opts.GridOpts(pos_bottom="85%", pos_right="78%"))
        )
        time_sd.add(grid, time_point=fea)

    # --------------------------加入时间线------------------------------------------------
    time_sd.add_schema(
        is_auto_play=False,  # 自启动
        play_interval=1000,
        is_inverse=True,  # 交换首尾
        pos_right='2',
        width=max([len(fea_list[i]) for i in range(len(fea_list))])*7,
        height=850,
        orient="vertical",
        symbol_size=8,
        label_opts=opts.series_options.LabelOpts(
            interval=0,
            font_size=11,
        ),
    )
    return time_sd


# --------------------------PCA降维------------------------------------------------
def mark_pca_sd(n_num, np_fea, fea_list):
    pca = PCA(n_components=2)  # n_components 选择降维数量
    pca = pca.fit(np_fea[fea_list])  # 开始拟合建模
    pca_data = pca.transform(np_fea[fea_list])  # 获得降维后数据
    pca_data = pd.DataFrame(pca_data, columns=['x', 'y'])
    pca_sd = (
        Scatter(
            init_opts=opts.InitOpts(theme=ThemeType.WESTEROS, chart_id='pca_sd'))
        .add_xaxis(list(pca_data['x']))
        .add_yaxis(
            series_name="阴性",
            y_axis=list(pca_data['y'])[:n_num],
            symbol_size=6,
            label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(color='#65c1e9')
        )
        .add_yaxis(
            series_name="阳性",
            y_axis=list(pca_data['y'])[n_num:],
            symbol_size=6,
            label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(color='#EC7063')

        )

        .set_global_opts(
            title_opts=opts.TitleOpts(title='PCA',
                                      pos_left='center',
                                      pos_top='5%',
                                      title_textstyle_opts=opts.TextStyleOpts(
                                          font_size=33, font_weight='bolder', color='#516b91')
                                      ),
            # xaxis_opts=opts.AxisOpts(is_show=False),
            # yaxis_opts=opts.AxisOpts(is_show=False,is_scale=True),
            legend_opts=opts.LegendOpts(pos_top='11%',),
            tooltip_opts=opts.TooltipOpts(
                formatter=JsCode(
                    "function (params) {return params.value[2];}"
                ),
                axis_pointer_type="cross",
            )

        )
    )
    return pca_sd
