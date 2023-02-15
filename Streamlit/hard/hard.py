import streamlit as st
import streamlit_echarts

from sklearn.model_selection import train_test_split
import pandas as pd

import io
import os
import json

import box
import draw
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


path="G:/item/工作/hard/"

#读取json
def open_json(file_name):
    with open(path+file_name+'.json','r+') as file:
        data_json=file.read()

    data_dict=json.loads(data_json)#转化为json格式文件
    return data_dict

#保存json
def sava_json(data_dict,file_name):
    data_json=json.dumps(data_dict)#转化为json格式文件

    with open(path+file_name+'.json','w+') as file:
        file.write(data_json)
#------------------------------------------初始化----------------------------------------------
st.set_page_config(page_title="Hard",page_icon="💎",layout="wide",initial_sidebar_state="expanded")

sysmenu = '''
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
'''
st.markdown(sysmenu,unsafe_allow_html=True)

#默认为空的缓存
sslist_false=['n_fea','p_fea','fea_list','np_fea','n_train','p_train','n_test','p_test','recommend_fea','config']
for ss in sslist_false:
    if ss not in st.session_state:
        st.session_state[ss] = None

fea_json=open_json("fea")
#---------------------------------------导航栏-----------------------------------------------------
with st.sidebar:
    expander1=st.expander('上传数据', expanded=False)
    n_uploaded_file = expander1.file_uploader("阴性",type=['csv'])
    p_uploaded_file = expander1.file_uploader("阳性",type=['csv'])


    if n_uploaded_file is not None:
        st.session_state['n_fea'] = get_df(n_uploaded_file)
        n_file_name = os.path.splitext(n_uploaded_file.name)
        st.session_state['n_file_name'] = n_file_name[0]

    if p_uploaded_file is not None:
        st.session_state['p_fea'] = get_df(p_uploaded_file)
        p_file_name = os.path.splitext(p_uploaded_file.name)
        st.session_state['p_file_name'] = p_file_name[0]

    if n_uploaded_file is not None and p_uploaded_file is not None:
        all_fea=list(st.session_state['n_fea'].columns)[1:]
        np_fea=pd.concat([st.session_state['n_fea'],st.session_state['p_fea']],axis=0)
        np_fea=np_fea[all_fea]
        target=[0 for i in range(len(st.session_state['n_fea']))]+[1 for i in range(len(st.session_state['p_fea']))]
        np_fea['target']=target

        st.session_state['np_fea'] = np_fea

#---------------------------------------选择特征-----------------------------------------------------
    expander2=st.expander('选择特征', expanded=False)
    tab1, tab2 = expander2.tabs(["选择特征", "批量更改"])
    def savafea_click(fea_dict):
        sava_json(fea_dict,"fea")
    with tab1:
        with st.form("fea_form1"):#选择
            st.markdown("###### 🔴 总的")
            all_fea = st.multiselect('',
            default = fea_json["all"],
            options = fea_json["own"],
            label_visibility ="collapsed",
            key="all_fea"
            )

            st.markdown("###### 🟡 海洛因")
            her_fea = st.multiselect('',
            default = fea_json["her"],
            options = fea_json["own"],
            label_visibility ="collapsed",
            key="her_fea"
            )

            st.markdown("###### 🔵 冰毒")
            ice_fea = st.multiselect('',
            default = fea_json["ice"],
            options = fea_json["own"],
            label_visibility ="collapsed",
            key="ice_fea"
            )
            col1,col2=st.columns(2)
            col1.form_submit_button("刷新")
            fea_dict={"all":all_fea,"her":her_fea,"ice":ice_fea}
            st.session_state['fea_dict']=fea_dict
            savafea_button=col2.form_submit_button("保存",on_click=savafea_click,args=(fea_dict,))
    with tab2:
        with st.form("fea_form2"):#批量
            st.markdown("###### 🔴 总的")

            all_fea = st.text_area('总的',value=fea_json["all"],label_visibility="collapsed")

            st.markdown("###### 🟡 海洛因")
            her_fea = st.text_area('海洛因',value=fea_json["her"],label_visibility="collapsed")

            st.markdown("###### 🔵 冰毒")
            ice_fea = st.text_area('冰毒',value=fea_json["ice"],label_visibility="collapsed")

            st.markdown("###### 🟣 所有特征")
            own_fea = st.text_area('所有特征',value=fea_json["own"],label_visibility="collapsed")

            fea_dict={"all":eval(all_fea),"her":eval(her_fea),"ice":eval(ice_fea),"own":eval(own_fea)}
            savafea_button=st.form_submit_button("保存",on_click=savafea_click,args=(fea_dict,))

#---------------------------------------划分数据-----------------------------------------------------
    with st.form("hua_form"):
        st.markdown("#### 🪟 划分数据")
        way = st.radio("类型：",('all', 'her', 'ice'),horizontal=True)
        proportion= st.slider('验证集占比：', 0.0, 0.9, 0.2,0.1)
        if st.form_submit_button("🔪 划 分"):
            fea_list=st.session_state['fea_dict'][way]
            st.session_state['fea_list']=fea_list
            if proportion<0.1: #不划分
                st.session_state['n_train']=st.session_state['n_fea'][fea_list]
                st.session_state['p_train']=st.session_state['n_fea'][fea_list]
                st.session_state['n_test']=st.session_state['n_fea'][fea_list]
                st.session_state['p_test']=st.session_state['p_fea'][fea_list]
            else:
                X=st.session_state['np_fea'][fea_list]
                Y=st.session_state['np_fea'][['target']]
                X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=proportion) #,random_state=0 取消随机
                XY_train=pd.concat([X_train,Y_train],axis=1)
                XY_test=pd.concat([X_test,Y_test],axis=1)
                st.session_state['n_train']=XY_train[XY_train['target']==0]
                st.session_state['p_train']=XY_train[XY_train['target']==1]
                st.session_state['n_test']=XY_test[XY_test['target']==0]
                st.session_state['p_test']=XY_test[XY_test['target']==1]
            st.write("阴性：训练集 "+str(len(st.session_state['n_train']))+"条 测试集 "+str(len(st.session_state['n_test']))+"条")
            st.write("阳性：训练集 "+str(len(st.session_state['p_train']))+"条 测试集 "+str(len(st.session_state['p_test']))+"条")
#---------------------------------------分箱-----------------------------------------------------
if st.session_state['n_train'] is not None:
    st1,st2=st.columns([1,2])
    with st.form("box_form"):
        st.markdown("###### 📦 离散分箱")
        tab1, tab2, tab3= st.tabs(["固定分箱", "自动分箱","本地分箱"])
        with tab1: #固定分箱
            t1,t2,t3=st.columns(3)
            ng= t1.slider('分箱组数', 0, 100, 10)
            option1 = t2.selectbox('分箱方式',('k-means','等距分箱','等频分箱'))
            number1 = t3.number_input('拟合项数',min_value=0,max_value=100,value=3,key='n1')
            divide1=st.form_submit_button(" ✂️ 分 箱 ")

        with tab2: #自动分箱
            t1,t2,t3=st.columns(3)
            option2= t1.selectbox('分箱方式',('最优分箱','CART算法','卡方检验','最优KS'))
            number2= t2.number_input('拟合项数',min_value=0,max_value=100,value=3,key='n2')
            divide2=st.form_submit_button("✂️ 分 箱")

        with tab3: #本地分箱
            box_uploaded_file= st.file_uploader("titlen",type=['json'],label_visibility='collapsed')
            divide3=st.form_submit_button("✴️ 读 取")

        if divide1:
            st.session_state['config']=box.divide(st.session_state['n_train'],st.session_state['p_train'],st.session_state['fea_list'],ng,number1,option1)
        if divide2:
            st.session_state['config']=box.best_divide(st.session_state['n_train'],st.session_state['p_train'],st.session_state['fea_list'],number2,option2)

        if divide3:
            box_uploaded_file = box_uploaded_file.read()
            st.session_state['config'] = json.loads(box_uploaded_file)

    if st.session_state['config'] is not None:
        st.success(st.session_state['config']["way"]+" 分箱成功！")
        expander=st.expander('🗃️ 分箱结果', expanded=False)

        expander.json(st.session_state['config'],expanded=False)#
        expander.download_button(
                label="下载结果",
                data=json.dumps(st.session_state['config']),
                file_name=st.session_state['config']["way"]+'.json',
                mime='application/json',
                    )

        with st.form("count_form"):
            st.markdown("###### 🕊️ 计算结果")
            form1,form2,form3,form4,form5,form6=st.columns([1,0.3,1,1,1,1])
            threshold = form1.number_input('概率阈值',min_value=0.00,max_value=1.00,step=0.01)

            is_fit=form1.checkbox('是否拟合',value=True)

            if form1.form_submit_button('计算'):
                n_num=box.jugde(st.session_state['n_test'],st.session_state['config'],threshold,is_fit)
                p_num=box.jugde(st.session_state['p_test'],st.session_state['config'],threshold,is_fit)

                jc=p_num/len(st.session_state['p_test'])
                wp=n_num/len(st.session_state['n_test'])
                jq=(len(st.session_state['n_test'])-n_num+p_num)/(len(st.session_state['n_test'])+len(st.session_state['p_test']))
                form3.metric(label="检出率", value=str(round(jc*100,2))+"%", delta=str(p_num)+"/"+str(len(st.session_state['p_test'])),delta_color="off")
                form4.metric(label="误判率", value=str(round(wp*100,2))+"%", delta=str(n_num)+"/"+str(len(st.session_state['n_test'])),delta_color="off")
                form5.metric(label="精确率",
                value=str(round(jq*100,2))+"%",
                delta=str(len(st.session_state['n_test'])-n_num+p_num)+"/"+str(len(st.session_state['n_test'])+len(st.session_state['p_test'])),delta_color="off")

                st.session_state['config']["threshold"]=threshold
                st.session_state['config']["rate"]={"检出率":jc,"误判率":wp,"精确率":jq}
                st.session_state['config']["is_fit"]=is_fit
        time_box=draw.mark_time_box(st.session_state['config'])
        streamlit_echarts.st_pyecharts(
            time_box,
            height=600,
            width='100%'
        )

else:
    st.image("https://s1.ax1x.com/2022/11/09/zSBtgS.png",width=700)