
import streamlit as st
from imblearn.over_sampling import SVMSMOTE,SMOTE,ADASYN,RandomOverSampler
import pandas as pd

@st.cache
def over_SVMSMOTE(np_fea,fea_list):
    svmsomte=SVMSMOTE(random_state=0)
    x_data_=np_fea[fea_list]
    y_data_=np_fea["target"]
    x_data,y_data=svmsomte.fit_resample(x_data_,y_data_)
    np_fea=pd.concat([x_data,y_data],axis=1)
    n_fea=np_fea[np_fea['target']==0]
    p_fea=np_fea[np_fea['target']==1]
    return n_fea,p_fea,np_fea

@st.cache
def over_SMOTE(np_fea,fea_list):
    somte=SMOTE(random_state=0)
    x_data_=np_fea[fea_list]
    y_data_=np_fea["target"]
    x_data,y_data=somte.fit_resample(x_data_,y_data_)
    np_fea=pd.concat([x_data,y_data],axis=1)
    n_fea=np_fea[np_fea['target']==0]
    p_fea=np_fea[np_fea['target']==1]
    return n_fea,p_fea,np_fea

@st.cache
def over_ADASYN(np_fea,fea_list):
    adasyn=ADASYN(random_state=0)
    x_data_=np_fea[fea_list]
    y_data_=np_fea["target"]
    x_data,y_data=adasyn.fit_resample(x_data_,y_data_)
    np_fea=pd.concat([x_data,y_data],axis=1)
    n_fea=np_fea[np_fea['target']==0]
    p_fea=np_fea[np_fea['target']==1]
    return n_fea,p_fea,np_fea

@st.cache
def over_RandomOverSampler(np_fea,fea_list):
    randomoversampler=RandomOverSampler(random_state=0)
    x_data_=np_fea[fea_list]
    y_data_=np_fea["target"]
    x_data,y_data=randomoversampler.fit_resample(x_data_,y_data_)
    np_fea=pd.concat([x_data,y_data],axis=1)
    n_fea=np_fea[np_fea['target']==0]
    p_fea=np_fea[np_fea['target']==1]
    return n_fea,p_fea,np_fea
