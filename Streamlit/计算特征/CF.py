import streamlit as st
import streamlit_echarts
import pandas as pd
import io
import os
import re

import SingleFlash_125_Features

from pyecharts.charts import Boxplot
import pyecharts.options as opts
from pyecharts.globals import ThemeType
#----------------------------------函数------------------------------------
@st.cache
def get_df(uploaded_file):
    try:
        uploaded_file = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        df = pd.read_csv(uploaded_file,encoding='utf-8')
    except:
        uploaded_file = io.StringIO(uploaded_file.getvalue().decode("gbk"))
        df = pd.read_csv(uploaded_file,encoding='gbk')
    return df
#----------------------------------初始化-----------------------------------------
st.set_page_config(page_title="计算特征",page_icon="⭐")#,layout="wide" 
hide_st_style = """
          <style>
          #MainMenu {visibility: hidden;}
          footer {visibility: hidden;}
          header {visibility: hidden;}
          theme {base:dark;}

          </style>
          """
st.markdown(hide_st_style, unsafe_allow_html=True)

if 'fea_df' not in st.session_state:
    st.session_state['fea_df'] = None

if 'state' not in st.session_state:
    st.session_state['state'] = 1
#-------------------------------------------------------------------------------

with st.expander('**🔵 上传数据**',expanded=True):
    def up_change ():
        st.session_state["fea_df"] = None
    uploaded_file = st.file_uploader("",type=['csv'],on_change=up_change)
    if uploaded_file is not None:
        data = get_df(uploaded_file)
        name_df=pd.DataFrame(list(data.iloc[:,0]),columns=["name"])
        try:
            data=data['瞳孔半径'].astype('str').str.split(',',expand=True)
            data.columns=[str(i) for i in data.columns]
        except:
            data=pd.DataFrame(data.iloc[:,1:127])
        data=data.astype('float')
        file_name = os.path.splitext(uploaded_file.name)
        file_name=file_name[0]


with st.expander('**🟡 选择特征**', expanded=False):
    st.write("")
    def select():
        st.session_state['state']=abs(st.session_state['state']-1)
    st.button('全选/反选',on_click=select)

    #---------------------基础特征--------------------
    fea_list0=[
        "maxindex",
        "maxvalue",
        "minindex",
        "minvalue",
        "retindex",
        "retvalue",
        ]
    options0 = st.multiselect('基础特征',fea_list0,fea_list0,disabled=True)

    # ---------------------基础特征衍生--------------------
    fea_list1=[
        "range1",
        "range2",
        "F1",
        "F2",
        "V1",
        "V2",
        "V_sum",
        ]
    # ---------------------面积特征--------------------
    fea_list2=[
        "S",
        "S_down",
        "S_ret",
        "S_low",
        "S_sum",
        ]
    # ---------------------角度特征--------------------
    fea_list3=[
        "maxang",
        "minang",
        "retang",
        "sumang",
        ]
    # ---------------------形态特征--------------------
    fea_list4=[
        "var",
        "std",
        "cv",
        "dmq",
        "dnp",
        "curvature",
        ]
    # ---------------------Kats特征--------------------
    fea_list5=[
        'lumpiness',
        'level_shift_size',
        'stability',
        'std1st_der',
        ]
    # ---------------------TsFresh特征--------------------
    fea_list6=[
        "mac",
        "asoc",
        ]
    # ---------------------傅里叶变换--------------------
    fea_list7 = [
        "fft_imag_3",
        "fft_imag_6",
        "fft_angle_3",
        "fft_angle_2",
        "fft_real_2",
        "fft_real_5",
        "fft_abs_1",
        "fft_abs_22",
    ]
    # ---------------------线性最小二乘回归--------------------
    fea_list8 = [
        "alt_stderr_min_50",
        "alt_stderr_mean_10",
        "alt_stderr_mean_5",
        "alt_stderr_max_5",
        "alt_stderr_min_5",
        "alt_stderr_max_10",
        "alt_rvalue_min_50",
    ]
    # ---------------------块平方和--------------------
    fea_list9 = [
        "erbc_num10_focus2",
        "erbc_num10_focus4"
    ]
    # ---------------------交叉功率谱密度--------------------
    fea_list10=[
        "swd_coeff_2",
    ]

    if st.session_state['state']:
        options1 = st.multiselect('基础特征衍生', fea_list1, fea_list1)
        options2 = st.multiselect('面积特征', fea_list2, fea_list2)
        options3 = st.multiselect('角度特征', fea_list3, fea_list3)
        options4 = st.multiselect('形态特征', fea_list4, fea_list4)
        options5 = st.multiselect('Kats特征', fea_list5, fea_list5)
        options6 = st.multiselect('TsFresh特征', fea_list6, fea_list6)
        options7 = st.multiselect('傅里叶变换', fea_list7, fea_list7)
        options8 = st.multiselect('线性最小二乘回归', fea_list8, fea_list8)
        options9 = st.multiselect('块平方和', fea_list9, fea_list9)
        options10 = st.multiselect('交叉功率谱密度', fea_list10, fea_list10)
    else:
        options1 = st.multiselect('基础特征衍生', fea_list1)
        options2 = st.multiselect('面积特征', fea_list2)
        options3 = st.multiselect('角度特征', fea_list3)
        options4 = st.multiselect('形态特征', fea_list4)
        options5 = st.multiselect('Kats特征', fea_list5)
        options6 = st.multiselect('TsFresh特征',fea_list6)
        options7 = st.multiselect('傅里叶变换', fea_list7)
        options8 = st.multiselect('线性最小二乘回归',fea_list8)
        options9 = st.multiselect('块平方和', fea_list9)
        options10 = st.multiselect('交叉功率谱密度', fea_list10)

    options_list = options1 + options2 + options3 + options4 + options5 + options6

with st.expander('**🔴 计算配置**', expanded=False):
    st.write("")
    col1,col2,col3=st.columns(3)
    is_pure1 = col1.checkbox('第一次去异常', value=True)
    is_pure2 = col2.checkbox('第二次去异常', value=True)
    is_smooth = col3.checkbox('平滑', value=True)
    is_round = col1.checkbox('保留小数', value=True)
    is_goone = col2.checkbox('归一化', value=True)
    is_pure3 = col3.checkbox('对特征值去异常', value=True)
    start_end = st.slider(label="始终帧数", min_value=0, max_value=124, value=(20, 101))

if st.button("**🔘 开始计算**"):
    with st.spinner('计算中...'):
        fea_all_dict={}
        for i in range(len(data)):
            fea_dict = {}
            row=list(data.iloc[i])
            Fea=SingleFlash_125_Features.Features(row,is_pure1=is_pure1, is_pure2=is_pure2, is_smooth=is_smooth, is_round=is_round, start=start_end[0], end=start_end[1])

            #基础特征
            for fea in options0:
                fea_dict[fea]=eval("Fea."+fea)
            #单个特征
            for fea in options_list:
                fea_dict[fea]=eval("Fea.get_"+fea+"()")

            #傅里叶
            if len(options7)!=0:
                fft_param=[]
                for fea in options7:
                    attr=re.findall("_(.*?)_", fea)[0]
                    coeff=int(re.findall("(\d+)",fea)[0])
                    fft_param.append({"coeff": coeff, "attr": attr})
                fea_dict.update(Fea.get_fft(fft_param))

            #线性最小二乘
            if len(options8) != 0:
                alt_param=[]
                for fea in options8:
                    attr=re.findall("alt_(.*?)_m", fea)[0]
                    chunk_len=int(re.findall("(\d+)", fea)[0])
                    f_agg=re.findall(attr+"_(.*?)_\d", fea)[0]
                    alt_param.append({"attr": attr, "chunk_len": chunk_len, "f_agg": f_agg})
                fea_dict.update(Fea.get_alt(alt_param))

            #块平方和
            if len(options9) != 0:
                erbc_param=[]
                for fea in fea_list9:
                    num_segments = int(re.findall("erbc_num(\d+)_", fea)[0])
                    segment_focus = int(re.findall("_focus(\d+)", fea)[0])
                    alt_param.append({"num_segments": num_segments, "segment_focus": segment_focus})
                fea_dict.update(Fea.get_erbc(erbc_param))

            #交叉功率谱密度
            if len(options10) != 0:
                swd_param = []
                for fea in fea_list10:
                    coeff=int(re.findall("(\d+)",fea)[0])
                    swd_param.append({"coeff": coeff})
                fea_dict.update(Fea.get_swd(swd_param))

            if is_pure3:
                fea_dict = SingleFlash_125_Features.pure3(fea_dict)
            if is_goone:
                fea_dict = SingleFlash_125_Features.goone(fea_dict)

            if i == 0:
                fea_list = list(fea_dict.keys())
                for fea in fea_list:
                    fea_all_dict[fea] = []
                    fea_all_dict[fea].append(fea_dict[fea])
            else:
                for fea in fea_list:
                    fea_all_dict[fea].append(fea_dict[fea])

        fea_df = pd.DataFrame(fea_all_dict)
        st.session_state["fea_df"] = pd.concat([name_df, fea_df], axis=1)
        st.success(file_name + " 特征计算完毕！", icon="✅")

if st.session_state["fea_df"] is not None:
    with st.expander('**🟢 数值详情**', expanded=False):
        st.write(st.session_state["fea_df"])
    with st.expander('**🟣 特征概况**', expanded=True):
        fea_df=st.session_state["fea_df"].copy()
        del fea_df['name']
        mark_data=[list(fea_df.iloc[:,i]) for i in range(fea_df.shape[1])]
        boxplot = Boxplot(init_opts=opts.InitOpts(theme=ThemeType.ROMA))
        boxplot.add_xaxis(list(fea_df.columns))
        boxplot.add_yaxis("",
                          boxplot.prepare_data(mark_data))  # , itemstyle_opts=opts.ItemStyleOpts(color='#fe8c7f')
        boxplot.set_global_opts(title_opts=opts.TitleOpts(title=""),
                                datazoom_opts=opts.DataZoomOpts(pos_bottom='bottom'),
                                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(interval=0,rotate=-45)),)
        streamlit_echarts.st_pyecharts(
            boxplot,
            height=420,
            width='100%'
        )
    st.download_button(
        label="**⬇️ 下载特征文件**",
        data=st.session_state["fea_df"].to_csv(index=False).encode('gbk'),
        file_name="[特征]"+file_name+'.csv',
        mime='text/csv',
    )
