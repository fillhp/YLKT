import streamlit as st
import io
import os
import pandas as pd
import numpy as np
#----------------------------------初始化-----------------------------------------
st.set_page_config(page_title="拟造数据",page_icon="💞")#,layout="wide" 
hide_st_style = """
          <style>
          #MainMenu {visibility: hidden;}
          footer {visibility: hidden;}
          header {visibility: hidden;}
          theme {base:dark;}

          </style>
          """
st.markdown(hide_st_style, unsafe_allow_html=True)

for var in ["new"]:
    if var not in st.session_state:
        st.session_state[var] = None
#----------------------------------函数-----------------------------------------
@st.cache
def get_df(uploaded_file):
    try:
        uploaded_file = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        df = pd.read_csv(uploaded_file,encoding='utf-8')
    except:
        uploaded_file = io.StringIO(uploaded_file.getvalue().decode("gbk"))
        df = pd.read_csv(uploaded_file,encoding='gbk')
    return df

@st.cache
def mean_clone(data,file_name,agree):
    new_dict={"name":[],"瞳孔半径":[]}
    for a in range(len(data)):
        row_a=np.array(data.iloc[a])
        for b in range(a+1,len(data)):
            name=file_name+'_'+str(a)+'_'+str(b)
            row_b=np.array(data.iloc[b])
            new_row=(row_a+row_b)/2
            if agree:
                noise = np.random.normal(0, np.mean(new_row)/100, len(new_row))
                new_row = list(new_row + noise)

            new_str=""
            for c in range(len(new_row)):
                if c!=len(new_row)-1:
                    new_str=new_str+str(new_row[c])+","
                else:
                    new_str=new_str+str(new_row[c])

            new_dict["name"].append(name)
            new_dict["瞳孔半径"].append(new_str)
    new=pd.DataFrame(new_dict)
    return new
#-----------------------------------body----------------------------
st.title('💞 拟造数据')
uploaded_file = st.file_uploader("上传原始数据",type=['csv'])
if uploaded_file is not None:
    data = get_df(uploaded_file)
    try:
        data=data['瞳孔半径'].astype('str').str.split(',',expand=True)
        data.columns=[str(i) for i in data.columns]
    except:
        data=pd.DataFrame(data.iloc[:,1:127])
    data=data.astype('float')
    file_name = os.path.splitext(uploaded_file.name)
    file_name=file_name[0]
    expander=st.expander('数据详情')#, expanded=True
    expander.write(data)
st.write("-"*50)

col1,col2=st.columns(2)
option = col1.selectbox('拟造方式',("均值拟造",""))
col2.write("")
col2.write("")
agree = col2.checkbox('是否随机增加噪音')

if st.button("拟造"):
    placeholder = st.empty()
    placeholder.image("https://i.imgtg.com/2022/12/09/DNAJb.gif")
    if option=="均值拟造":
        st.session_state["new"]=mean_clone(data,file_name,agree)

    if st.session_state["new"] is not None:
        col1,col2=st.columns(2)
        col1.download_button(
            label="下载",
            data=st.session_state["new"].to_csv(index=False).encode('gbk'),
            file_name="[拟造]"+file_name+'.csv',
            mime='text/csv',
        )
        col2.metric("拟造后数据", len(st.session_state["new"]), str(len(st.session_state["new"])/len(data))+"倍")
        with placeholder.container():
            expander=placeholder.expander('数据详情')#, expanded=True
            expander.write(st.session_state["new"])
