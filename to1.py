from random import sample
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
import streamlit as st
import requests
from streamlit_lottie import st_lottie
import joblib
st.set_page_config(page_title='stroke prediction', page_icon='::star::')


model = joblib.load(open("stroke prediction", 'rb'))

def load_lottie(url):
    if not url:
        return None
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def prepare_input_data_for_model(id,gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi):
    #sex = gender.map(gen)
    if gender == 'Male':
        gender = 0
    else:
        gender = 1
    #s_b = ssc_b.map(sb)
    if ever_married== 'Yes':
        ever_married = 1
    else:
        ever_married = 0
    #h_b = hsc_b.map(hb)
    if work_type == 'Private':
        work_type = 0
    elif work_type=='employed':
        work_type = 1
    elif work_type =='children':
        work_type = 2
    elif work_type =='Govt_job':
      work_type=3
    else:
      work_type=4

    #h_s = hsc_subject.map(h_sub)
    if Residence_type == 'Urban':
        Residence_type = 1
    else  :
        Residence_type = 0
    #if smoking_status=="never smoked":
     #smoking_status=0
    #else:
#smoking_status=1
   # prediction = model.predict(pd.DataFrame([[id,gender,age,hypertension,heart_disease,ever_married,Residence_type,work_type,avg_glucose_level,bmi,smoking_status]], columns=['id', 'age', 'gender', 'hypertension', 'heart_disease','ever_married', 'Residence_type', 'work_type', 'Residence_typ','avg_glucose_level','bmi','smoking_status']))
    prediction = model.predict(pd.DataFrame([[id,gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi]], columns=['id', 'gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi']))
    return prediction




st.write('# stroke prediction Deployment')
#st.header('Placement')

st.write('---')
st.subheader('troke data  predection')

lottie_link = "https://assets8.lottiefiles.com/packages/lf20_ax5yuc0o.json"
animation = load_lottie(lottie_link)

with st.container():

    right_column, left_column = st.columns(2)
#def prepare_input_data_for_model(id,gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status):

    with right_column:
        id = st.number_input('id:')


        gender = st.selectbox('Gender : ',['Male', 'Female'])
        age = st.number_input(' age: ', )

        hypertension= st.number_input('hypertension : ', )

        heart_disease = st.number_input('heart_disease : ', )
        
        ever_married = st.selectbox('ever married : ',['Yes','No'] )
        
        work_type = st.selectbox('work type  : ', ['private', 'employed', 'Govt_job','children'])

        Residence_type= st.selectbox('residence type : ',['Urban','Rural'])

        avg_glucose_level= st.number_input(' avg_gulcose_level: ', )

        bmi= st.number_input('bmi : ', )
        #smoking_status=st.selectbox('s_k : ',['never smoked','else'] )
        #smoking_status = st.selectbox('smoking status : ', ['','',,'',''])

    with left_column:
        st_lottie(animation, speed=1, height=400, key="initial")

        #df=pd.DataFrame([sample], columns=["id",'sex','age','hypertension','heart_disease','e_m','w_t','r_t','avg_glucose_level','bmi'])
    if st.button('Predict'):
            pred_Y=  prepare_input_data_for_model(id,gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi)
            print(pred_Y)
            if pred_Y == 0:
                #st.write("## Predicted Status : ", result)
                st.write('### Congratulations ', id, '!! You arenot patient.')
                st.balloons()
            else:
                st.write('### Sorry ', id, '!! You are patient.')


                