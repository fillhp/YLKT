import streamlit as st
from streamlit_option_menu import option_menu
import streamlit_echarts

import pandas as pd

import io
import os

import grade
import draw
import over
#------------------------------------------初始化----------------------------------------------
st.set_page_config(page_title="特征分析",page_icon="🌠",layout="wide",initial_sidebar_state="expanded")

sysmenu = '''
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
'''
st.markdown(sysmenu,unsafe_allow_html=True)

#默认为空的缓存
for ss in ['n_fea','p_fea','n_file_name','p_file_name','np_fea','screen_fea','recommend_fea','syn_list','old','new','init']:
    if ss not in st.session_state:
        st.session_state[ss] = None

# 默认为FALSE的缓存
for ss in ['is_options_fea','is_apply']:
    if ss not in st.session_state:
        st.session_state[ss] = False

for ss in ['all_fea','options_fea']:
    if ss not in st.session_state:
        st.session_state[ss] = ['请先上传特征数据']


#---------------------------------------函数-----------------------------------------------------
#获取文件
@st.cache
def get_df(uploaded_file):
    try:
        uploaded_file = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        df = pd.read_csv(uploaded_file,encoding='utf-8')
    except:
        uploaded_file = io.StringIO(uploaded_file.getvalue().decode("gbk"))
        df = pd.read_csv(uploaded_file,encoding='gbk')
    return df

#筛选
@st.cache
def screen(np_fea,fea_list,threshold,score_dict=None):
    np_fea=np_fea[fea_list]
    cor=np_fea.iloc[:,:].corr()
    screen_fea=fea_list.copy()
    for a in range(len(fea_list)):
        fea_a=fea_list[a]
        col=cor[fea_a]
        for b in range(len(fea_list)):
            fea_b=fea_list[b]
            if fea_a==fea_b:
                continue
            y=abs(col[fea_b])
            if y>=threshold:
                if score_dict==None:#直接筛选
                    if fea_b in screen_fea:
                        screen_fea.remove(fea_b)
                else:
                    if score_dict[fea_a]>score_dict[fea_b]:
                        if fea_b in screen_fea:
                            screen_fea.remove(fea_b)
                    else:
                        if fea_a in screen_fea:
                            screen_fea.remove(fea_a)
    return screen_fea
#---------------------------------------导航栏-----------------------------------------------------
with st.form("my_form"):
    st.markdown("###### 💘 选择特征")
    options_fea = st.multiselect(
    '选择特征',
    default = st.session_state['options_fea'],
    options = st.session_state['all_fea'],
    label_visibility ="collapsed"
    )

    form1,form2,form3,form4=st.columns([1,1,6,2])
    renovate = form1.form_submit_button("🌟 刷新选择")
    def restore_on_click():
        st.session_state['options_fea']=st.session_state['all_fea']
    restore= form2.form_submit_button("💫 还原全选",on_click=restore_on_click)

    with form4.expander("列表格式"):
        st.write(options_fea)
st.write(' ')#间隔


with st.sidebar:
    choose = option_menu("特征分析", ["上传数据","过采样","稳定性", "重要性", "相似度", "散点图","PCA"],
                         icons=['justify-left', 'plus-circle-dotted','triangle','patch-check', 'heart-half', 'dice-5','nut'],
                         menu_icon="star-fill", default_index=0)

    if st.session_state['np_fea'] is not None:
        st.success("🌙 "+st.session_state['n_file_name']+' - '+str(len(st.session_state['n_fea']))+' 条数据')
        st.error("☀️ "+st.session_state['p_file_name']+' - '+str(len(st.session_state['p_fea']))+' 条数据')

#---------------------------------------选择数据-----------------------------------------------------
if choose == "上传数据":
    col1, col2 = st.columns(2)
    col1.markdown("###### 🌙 请选择阴性特征数据")
    n_uploaded_file = col1.file_uploader("titlen",type=['csv'],label_visibility='collapsed')

    col2.markdown("###### ☀️ 请选择阳性特征数据")
    p_uploaded_file = col2.file_uploader("titlep",type=['csv'],label_visibility='collapsed')


    if n_uploaded_file is not None:
        st.session_state['n_fea'] = get_df(n_uploaded_file)
        n_file_name = os.path.splitext(n_uploaded_file.name)
        st.session_state['n_file_name'] = n_file_name[0]

    if p_uploaded_file is not None:
        st.session_state['p_fea'] = get_df(p_uploaded_file)
        p_file_name = os.path.splitext(p_uploaded_file.name)
        st.session_state['p_file_name'] = p_file_name[0]

    if st.session_state['n_fea'] is not None:
        col1.success(st.session_state['n_file_name']+' - '+str(len(st.session_state['n_fea']))+' 条数据')
        expander1=col1.expander('数据详情', expanded=True)
        expander1.write(st.session_state['n_fea'])

    if st.session_state['p_fea'] is not None:
        col2.error(st.session_state['p_file_name']+' - '+str(len(st.session_state['p_fea']))+' 条数据')
        expander2=col2.expander('数据详情', expanded=True)
        expander2.write(st.session_state['p_fea'])

    if n_uploaded_file is not None and p_uploaded_file is not None:
        all_fea=list(st.session_state['n_fea'].columns)[1:]
        np_fea=pd.concat([st.session_state['n_fea'],st.session_state['p_fea']],axis=0)
        np_fea=np_fea[all_fea]
        target=[0 for i in range(len(st.session_state['n_fea']))]+[1 for i in range(len(st.session_state['p_fea']))]
        np_fea['target']=target

        st.session_state['np_fea'] = np_fea
        st.session_state['all_fea'] = all_fea
        st.session_state['options_fea']=all_fea

#---------------------------------------过采样-----------------------------------------------------
elif choose == "过采样":
    if st.session_state['np_fea'] is not None:
        with st.form("over_form"):
            col1,col2,col3,col4,col5,col6,col7=st.columns([3,1,1,1,1,1,2])
            over_way = col1.selectbox(
                '过采样方法',
                ('SMOTE','SVMSMOTE','ADASYN','RandomOverSampler'))

            if st.session_state['is_apply']:
                st.error("过采样结果应用成功！")
                def back_click():
                    if st.session_state['init']['n'][0] < st.session_state['init']['p'][0]:
                        st.session_state['n_fea'] = st.session_state['old']

                    else:
                        st.session_state['p_fea'] = st.session_state['old']
                    st.session_state['old'] = None
                    st.session_state['is_apply'] = False

                col3.write('')
                col3.write('')
                col3.form_submit_button("🔙 还原", on_click=back_click)

            col2.write('')
            col2.write('')
            if col2.form_submit_button("🔘 过采样"):
                if len(st.session_state['n_fea'])==len(st.session_state['p_fea']):
                    st.info("阴性和阳性数据量已平衡，无需过采样")
                else:
                    ret=eval("over.over_"+over_way+"(st.session_state['np_fea'], options_fea)")
                    if len(st.session_state['n_fea'])<len(st.session_state['p_fea']):
                        st.session_state['old'] = st.session_state['n_fea']
                        st.session_state['new'] = ret[0].iloc[len(st.session_state['n_fea']):]
                    else:
                        st.session_state['old'] = st.session_state['p_fea']
                        st.session_state['new'] = ret[1].iloc[len(st.session_state['p_fea']):]

                    st.session_state['init'] = {'n':(len(st.session_state['n_fea']),len(ret[0])),
                                                'p':(len(st.session_state['p_fea']),len(ret[1])),
                                                'np':(len(st.session_state['np_fea']),len(ret[2]))}
                    col3.write('')
                    col3.write('')
                    def apply_click():
                        st.session_state['n_fea'] = ret[0]
                        st.session_state['p_fea'] = ret[1]
                        st.session_state['np_fea'] = ret[2]
                        st.session_state['is_apply']=True
                    col3.form_submit_button("✔️ 应用",on_click=apply_click)

            if st.session_state['old'] is not None:
                    boxplot = draw.mark_boxplot(st.session_state['new'], st.session_state['old'],options_fea,
                                                tit1="过采样数据",tit2="原数据",  tit3="过采样箱型图")

                    streamlit_echarts.st_pyecharts(
                        boxplot,
                        height=500,
                        width='100%'
                    )



                    col4.metric("阴性数据", str(st.session_state['init']['n'][1]) + "条",
                                str(round(st.session_state['init']['n'][1] * 100 / st.session_state['init']['n'][0])) + "%")

                    col5.metric("阳性数据", str(st.session_state['init']['p'][1]) + "条",
                                str(round(st.session_state['init']['p'][1] * 100 / st.session_state['init']['p'][0])) + "%")
                    col6.metric("总数据", str(st.session_state['init']['np'][1]) + "条",
                                str(round(st.session_state['init']['np'][1] * 100 / st.session_state['init']['np'][0])) + "%")

    else:
        st.error("请先上传特征数据！")
#---------------------------------------稳定性-----------------------------------------------------
elif choose == "稳定性":
    if st.session_state['np_fea'] is not None:
        std_rank_chart=draw.mark_std_rank(st.session_state['n_fea'],st.session_state['p_fea'],options_fea)
        streamlit_echarts.st_pyecharts(
            std_rank_chart,
            height=300,
            width='100%'
        )

        boxplot=draw.mark_boxplot(st.session_state['p_fea'],st.session_state['n_fea'],options_fea,tit1="阳性",tit2="阴性",tit3="特征值箱型图")
        streamlit_echarts.st_pyecharts(
            boxplot,
            height=500,
            width='100%'
        )
    else:
        st.error("请先上传特征数据！")
elif choose == "重要性":
    if st.session_state['np_fea'] is not None:
        container = st.container()#from占位

        col1,col2,col3,col4,col5=st.columns(5)

        col1.markdown('###### 🍕 基于差值')
        gap_dict=grade.gap(st.session_state['n_fea'],st.session_state['p_fea'],options_fea)
        col1.write(gap_dict)

        col2.markdown('###### 🪨 基于硬算')
        hard_dict=grade.hard(st.session_state['np_fea'],options_fea)
        col2.write(hard_dict)

        col3.markdown('###### 🌲 基于随机森林')
        rf_dict=grade.rf(st.session_state['np_fea'],options_fea)
        col3.write(rf_dict)

        col4.markdown('###### 🌴 基于极度随机树')
        etc_dict=grade.etc(st.session_state['np_fea'],options_fea)
        col4.write(etc_dict)

        # col4.markdown('###### 🫧 基于xgboost')
        # xgboost_dict=grade.xgboost(st.session_state['np_fea'],options_fea)
        # col4.write(xgboost_dict)

        # col4.markdown('###### 🎰 基于IV评分卡')
        # iv_dict=grade.iv(st.session_state['np_fea'],options_fea)
        # col4.write(iv_dict)

        col5.markdown('###### 🎯 综合排名')
        syn=grade.syn(gap_dict,hard_dict,rf_dict,etc_dict)
        syn_list=syn[0]
        st.session_state['syn_list']=syn_list
        loser_list=syn[1]
        col5.write(syn_list)
        col5.markdown('###### 🗑️ 无影响力特征')
        col5.write(loser_list)

        with container.form("recommend_form"):
            st.markdown("###### 🧨 推荐特征")
            col1,col2,col3,col4,col5=st.columns([3,0.5,3,0.5,1])
            threshold = col1.slider('推荐阈值', 0, 100, 50)
            option = col3.selectbox('推荐方式',('考虑综合排名','考虑硬算评分','考虑差值评分', '考虑随机森林评分','考虑极度随机树评分'))
            screen_fea=0
            if col5.form_submit_button("🔘 推 荐"):
                st.session_state['recommend_fea']=grade.recommend(gap_dict,hard_dict,rf_dict,etc_dict,syn_list,loser_list,option,threshold)

                st.markdown("###### 🎉 推荐结果：")
                st.info(st.session_state['recommend_fea'])

            def apply_on_click(recommend_fea):
                st.session_state['options_fea']=recommend_fea
            apply_button=col5.form_submit_button("✔️ 应 用",on_click=apply_on_click,args=([st.session_state['recommend_fea']]))
    else:
        st.error("请先上传特征数据！")
#---------------------------------------相似度-----------------------------------------------------
elif choose == "相似度":
    if st.session_state['np_fea'] is not None:
        with st.form("resemble_form"):
            st.markdown("###### 🧨 相似度筛选")
            col1,col2,col3,col4,col5=st.columns([3,0.5,3,0.5,1])
            threshold = col1.slider('筛选阈值', 0.0, 1.0, 0.88)
            if st.session_state['syn_list'] is None:
                option = col3.selectbox('筛选方式',('考虑差值评分','考虑硬算评分','考虑随机森林评分','考虑极度随机树评分','直接筛选'))
            else:
                option = col3.selectbox('筛选方式',('考虑综合排名', '考虑差值评分','考虑硬算评分','考虑随机森林评分','考虑极度随机树评分','直接筛选'))
            screen_fea=0
            if col5.form_submit_button("🔘 筛 选"):
                if option=="考虑综合排名":
                    score_dict=dict(zip(st.session_state['syn_list'],[i for i in range(len(st.session_state['syn_list']),0,-1)]))
                    st.session_state['screen_fea']=screen(st.session_state['np_fea'],options_fea,threshold,score_dict)
                elif option=="考虑差值评分":
                    score_dict=grade.gap(st.session_state['n_fea'],st.session_state['p_fea'],options_fea)
                    st.session_state['screen_fea']=screen(st.session_state['np_fea'],options_fea,threshold,score_dict)
                elif option=="考虑硬算评分":
                    score_dict=grade.hard(st.session_state['np_fea'],options_fea)
                    st.session_state['screen_fea']=screen(st.session_state['np_fea'],options_fea,threshold,score_dict)
                elif option=="考虑随机森林评分":
                    score_dict=grade.rf(st.session_state['np_fea'],options_fea)
                    st.session_state['screen_fea']=screen(st.session_state['np_fea'],options_fea,threshold,score_dict)
                elif option=="考虑极度随机树评分":
                    score_dict=grade.etc(st.session_state['np_fea'],options_fea)
                    st.session_state['screen_fea']=screen(st.session_state['np_fea'],options_fea,threshold,score_dict)
                # elif option=="xgboost":
                #     score_dict=grade.xgboost(st.session_state['np_fea'],options_fea)
                #     st.session_state['screen_fea']=screen(st.session_state['np_fea'],options_fea,threshold,score_dict)
                # elif option=="考虑IV评分":
                #     score_dict=grade.iv(st.session_state['np_fea'],options_fea)
                #     st.session_state['screen_fea']=screen(st.session_state['np_fea'],options_fea,threshold,score_dict)
                elif option=="直接筛选":
                    st.session_state['screen_fea']=screen(st.session_state['np_fea'],options_fea,threshold)
                st.markdown("###### 🎉 筛选结果：")
                st.info(st.session_state['screen_fea'])

            def apply_on_click(screen_fea):
                st.session_state['options_fea']=screen_fea
            apply_button=col5.form_submit_button("✔️ 应 用",on_click=apply_on_click,args=([st.session_state['screen_fea']]))


        hat=draw.mark_hat(st.session_state['np_fea'],options_fea)
        streamlit_echarts.st_pyecharts(
            hat,
            height=1000,
            width='100%'
        )

    else:
        st.error("请先上传特征数据！")
#---------------------------------------散点图-----------------------------------------------------
elif choose == "散点图":
    if st.session_state['np_fea'] is not None:
        if st.session_state['old'] is not None:
            if st.session_state['init']['n'][0] < st.session_state['init']['p'][0]:
                time_sd=draw.mark_time_sd(st.session_state['old'],st.session_state['p_fea'],st.session_state['new'],0,options_fea)
            else:
                time_sd=draw.mark_time_sd(st.session_state['n_fea'],st.session_state['old'],st.session_state['new'],1,options_fea)
        else:
            time_sd = draw.mark_time_sd(st.session_state['n_fea'], st.session_state['p_fea'],
                                        st.session_state['new'], 2, options_fea)

        streamlit_echarts.st_pyecharts(
            time_sd,
            height=850,
            width='100%'
        )
    else:
        st.error("请先上传特征数据！")
#---------------------------------------PCA-----------------------------------------------------
elif choose == "PCA":
    if st.session_state['np_fea'] is not None:
        pca_sd=draw.mark_pca_sd(len(st.session_state['n_fea']),st.session_state['np_fea'],options_fea)
        streamlit_echarts.st_pyecharts(
            pca_sd,
            height=600,
            width='100%'
        )
    else:
        st.error("请先上传特征数据！")