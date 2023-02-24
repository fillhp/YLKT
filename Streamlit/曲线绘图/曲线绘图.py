# ---------------------------------导入库------------------------------------------
import io

import pandas as pd
import pyecharts.options as opts
import streamlit as st
import streamlit_echarts
from pyecharts.charts import Line, EffectScatter
from pyecharts.commons.utils import JsCode

import SingleFlash_125_Features


# ---------------------------------数据处理函数------------------------------------------

@st.cache_data
def get_df(uploaded_file):
    try:
        uploaded_file = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except:
        uploaded_file = io.StringIO(uploaded_file.getvalue().decode("gbk"))
        df = pd.read_csv(uploaded_file, encoding='gbk')
    return df


# ---------------------------------基础绘图函数------------------------------------------
def draw(pure_data, name_list, line_show, tooltip_show, line_width, change_color, change_color_add, y_genre, ini,draw_tit):
    if y_genre == "纵坐标从最小值开始":
        y_min_ = 'dataMin'
    elif y_genre == "纵坐标从0开始":
        y_min_ = 0
    Line_chart = (
        Line()
        .add_xaxis([i for i in range(125)])
        .set_global_opts(
            title_opts=opts.TitleOpts(title="共" + str(draw_tit[0]) + '条数据',
                                      subtitle="正常数据："+str(draw_tit[1])+" 异常数据："+str(draw_tit[2]),
                                      title_textstyle_opts=opts.TextStyleOpts(
                                          font_size=25, font_weight='bolder')
                                      ),
            legend_opts=opts.LegendOpts(type_="scroll", pos_top='top',pos_right="right", orient="vertical"),  # 图例
            tooltip_opts=opts.TooltipOpts(
                trigger="axis",
                formatter='{c}',
                is_show=tooltip_show
            ),  # 竖标注线
            yaxis_opts=opts.AxisOpts(
                min_=y_min_,  # 从最小值开始画图
                interval=2,  # y轴间隔
            ),
            xaxis_opts=opts.AxisOpts(
                interval=5,  # x轴间隔
                max_=115 + len(name_list[0]),
            ),
        )  # set_global_opts的括号
    )

    for i in range(len(pure_data)):
        if i >= ini:
            change_color = change_color_add
        Line_chart.add_yaxis(
            name_list[i]
            , pure_data[i]
            , label_opts=opts.LabelOpts(is_show=False)
            , is_symbol_show=False
            , is_smooth=False
            , is_selected=line_show  # 图例全部取消选中
            , linestyle_opts=opts.LineStyleOpts(width=line_width)
            , itemstyle_opts=opts.ItemStyleOpts(color=change_color)
        )

    return Line_chart


# ---------------------------------特征绘图函数------------------------------------------
def draw_fea(name_list, peak_data, line_width, fea_value_show, line_show, change_color, change_color_add, ini):
    sd_fea = (
        EffectScatter()
        .add_xaxis([i for i in range(125)])
        .set_global_opts(
            yaxis_opts=opts.AxisOpts(is_show=False),
            xaxis_opts=opts.AxisOpts(is_show=False),
            tooltip_opts=opts.TooltipOpts(
                is_show=False,
            )
        )
    )
    formatter = JsCode(
        """
        function (params) {
            return '('+ params.value[0] +','+params.value[1].toFixed(0) + ')';
        }
        """
    )
    for i in range(len(peak_data)):
        if i >= ini:
            change_color = change_color_add
        row = peak_data.iloc[i]
        name = name_list[i]
        max_list = [None for j in range(125)]
        min_list = max_list.copy()
        ret_list = max_list.copy()

        max_list[int(row['maxindex'])] = row['maxvalue']
        min_list[int(row['minindex'])] = row['minvalue']
        ret_list[int(row['retindex'])] = row['retvalue']

        sd_fea.add_yaxis(name, max_list, symbol_size=line_width * 6,
                         label_opts=opts.LabelOpts(is_show=fea_value_show, formatter=formatter)
                         , is_selected=line_show, itemstyle_opts=opts.ItemStyleOpts(color=change_color))

        sd_fea.add_yaxis(name, min_list, symbol_size=line_width * 6,
                         label_opts=opts.LabelOpts(is_show=fea_value_show, formatter=formatter)
                         , is_selected=line_show, itemstyle_opts=opts.ItemStyleOpts(color=change_color))

        sd_fea.add_yaxis(name, ret_list, symbol_size=line_width * 6,
                         label_opts=opts.LabelOpts(is_show=fea_value_show, formatter=formatter)
                         , is_selected=line_show, itemstyle_opts=opts.ItemStyleOpts(color=change_color))


    return sd_fea


# ---------------------------------初始化------------------------------------------
st.set_page_config(page_title="曲线绘图", page_icon="📈", layout="wide")
hide_st_style = """
          <style>
          #MainMenu {visibility: hidden;}
          footer {visibility: hidden;}
          header {visibility: hidden;}
          </style>
          """
st.markdown(hide_st_style, unsafe_allow_html=True)

# ---------------------------------侧边栏------------------------------------
st.sidebar.header("📌 上传数据")
uploaded_file = st.sidebar.file_uploader("请选择数据", type=['csv'], label_visibility='collapsed')

st.sidebar.write('-' * 50)  # 分割线------------------------
sidebar_left, sidebar_right = st.sidebar.columns(2)
is_pure1 = sidebar_left.checkbox('第一次去异常', value=True)
is_pure2 = sidebar_right.checkbox('第二次去异常', value=True)
is_smooth = sidebar_left.checkbox('数据平滑', value=True)
original_show = sidebar_right.checkbox('与原始数据对比', value=False)
pure_num = st.sidebar.number_input('去异常次数上限', value=10)

st.sidebar.write('-' * 50)  # 分割线------------------------
sidebar_left, sidebar_right = st.sidebar.columns(2)
tooltip_show = sidebar_left.checkbox('标注线显示', value=True)
line_show = sidebar_right.checkbox('曲线全选/反选', value=True)
fea_show = sidebar_left.checkbox('极值点显示', value=True)
fea_value_show = sidebar_right.checkbox('极值坐标显示', value=False)

y_genre = st.sidebar.radio(
    "What\'s your favorite movie genre",
    ('纵坐标从0开始', '纵坐标从最小值开始'), horizontal=True, label_visibility="collapsed")

st.sidebar.write('-' * 50)  # 分割线------------------------
exp_genre = st.sidebar.radio(
    "What\'s your favorite movie genre",
    ('所有数据', '正常数据', '异常数据'), horizontal=True, label_visibility="collapsed")
exception_threshold = st.sidebar.slider('检测异常阈值', min_value=0.0, max_value=1.0, value=0.5, step=0.1)

st.sidebar.write('-' * 50)  # 分割线------------------------
line_width = st.sidebar.slider('线条宽度', min_value=0.5, max_value=2.0, value=1.0, step=0.1)

sidebar_left, sidebar_right = st.sidebar.columns([2, 1])
change = sidebar_left.checkbox('统一改变线条颜色', value=False)
change_color = None
if change:
    change_color = sidebar_right.color_picker('', '#3AB6E8', label_visibility='collapsed')
else:
    change_color = None

# ---------------------------------可视化区域------------------------------------
if uploaded_file is not None:
    df = get_df(uploaded_file)
    ini = len(df)  # 初始数据的数目
    # 追加数据----------------
    expander = st.sidebar.expander("追加数据")
    uploaded_file_add = expander.file_uploader("请选择追加数据", type=['csv'], label_visibility='collapsed')
    change_color_add = None
    if uploaded_file_add is not None:
        df_add = get_df(uploaded_file_add)
        df = pd.concat([df, df_add], axis=0)

        expander_left, expander_right = expander.columns([2, 1])
        change_add = expander_left.checkbox('改变追加线条颜色', value=False)

        if change_add:
            change_color_add = expander_right.color_picker('', '#E7593C', label_visibility='collapsed')
        else:
            change_color_add = None

    df.iloc[:, 0].astype(str)
    name_list = list(df.iloc[:, 0])

    data = df['瞳孔半径'].astype('str').str.split(',', expand=True)
    data.columns = [str(i) for i in data.columns]  # 修改分割后的字段名称
    data = data.astype('float')

    peak_dict = {'maxindex': [], 'maxvalue': [], 'minindex': [], 'minvalue': [], 'retindex': [], 'retvalue': []}
    pure_data = []
    selected_list = []
    new_name_list = []

    draw_tit=[len(data),0,0] #总数据，正常数据，异常数据
    for i in range(len(data)):
        row = list(data.iloc[i])
        Fea = SingleFlash_125_Features.Features(row, is_pure1, is_pure2, pure_num, exception_threshold, is_smooth)

        if exp_genre == "所有数据":
            selected_list.append(data.index[i])
            if Fea.exception:
                new_name_list.append(name_list[i] + "[异常]")
                draw_tit[2]+=1
            else:
                new_name_list.append(name_list[i])
                draw_tit[1]+=1

            pure_data.append(Fea.row)
            peak_dict['maxindex'].append(Fea.maxindex)
            peak_dict['maxvalue'].append(Fea.maxvalue)
            peak_dict['minindex'].append(Fea.minindex)
            peak_dict['minvalue'].append(Fea.minvalue)
            peak_dict['retindex'].append(Fea.retindex)
            peak_dict['retvalue'].append(Fea.retvalue)

        elif exp_genre == "正常数据":
            if Fea.exception == False:
                new_name_list.append(name_list[i])
                selected_list.append(data.index[i])
                draw_tit[1]+=1
                pure_data.append(Fea.row)
                peak_dict['maxindex'].append(Fea.maxindex)
                peak_dict['maxvalue'].append(Fea.maxvalue)
                peak_dict['minindex'].append(Fea.minindex)
                peak_dict['minvalue'].append(Fea.minvalue)
                peak_dict['retindex'].append(Fea.retindex)
                peak_dict['retvalue'].append(Fea.retvalue)
            else:
                draw_tit[2]+=1
        elif exp_genre == "异常数据":
            if Fea.exception:
                draw_tit[2]+=1
                new_name_list.append(name_list[i])
                selected_list.append(data.index[i])
                pure_data.append(Fea.row)
                peak_dict['maxindex'].append(Fea.maxindex)
                peak_dict['maxvalue'].append(Fea.maxvalue)
                peak_dict['minindex'].append(Fea.minindex)
                peak_dict['minvalue'].append(Fea.minvalue)
                peak_dict['retindex'].append(Fea.retindex)
                peak_dict['retvalue'].append(Fea.retvalue)
            else:
                draw_tit[1]+=1

    peak_data = pd.DataFrame(peak_dict)
    if len(new_name_list) != 0:
        line = draw(pure_data, new_name_list, line_show, tooltip_show, line_width, change_color, change_color_add,
                    y_genre, ini,draw_tit)  # 基础曲线

    old_data = data.loc[selected_list]
    if original_show and len(new_name_list) != 0:  # 与原数据对比
        for i in range(len(old_data)):
            line.add_yaxis(
                new_name_list[i]
                , list(old_data.iloc[i])
                , label_opts=opts.LabelOpts(is_show=False)
                , is_symbol_show=False
                , is_smooth=False
                , is_selected=line_show  # 图例全部取消选中
                , linestyle_opts=opts.LineStyleOpts(width=line_width, color='gary')
            )

    if fea_show and len(new_name_list) != 0:  # 最大值 最小值 返回值
        sd_fea = draw_fea(new_name_list, peak_data, line_width, fea_value_show, line_show, change_color,
                          change_color_add, ini)
        line.overlap(sd_fea)

    if len(new_name_list) != 0:
        streamlit_echarts.st_pyecharts(
            line,
            height='900%',
            width='100%'
        )
    else:
        st.error("无数据！")
