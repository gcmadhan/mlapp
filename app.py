import streamlit as st
from streamlit_pandas_profiling import st_profile_report
#import pandas_profiling
from pycaret.classification import *
from sklearn.datasets import load_iris


import pandas as pd
import os

if os.path.exists("source.csv"):
    df=pd.read_csv("source.csv", index_col=None)

if os.path.exists("model"):
    model=load_model('model')

st.title("Classification Model")
with st.sidebar:
    opt  = st.radio("Choose the step below:",("Upload file", "Data Analysis", "Modeling","Predict"))
    st.write(opt)

if opt == "Upload file":
    fs = st.selectbox("Choose data source",("Upload file","Use Sample data"))
    if fs =="Upload file":
        file = st.file_uploader("Upload CSV file: ", type=["csv"], accept_multiple_files=False)
        if file:
            #st.write(file.type)
            df = pd.read_csv(file, index_col=None)
            st.dataframe(df)
            df.to_csv("source.csv", index=None)
    elif fs =="Use Sample data":
        data = load_iris(as_frame=True, return_X_y=False)
        df = pd.DataFrame(data=data.data, columns=data.feature_names)
        df['target']=pd.Series(data.target)
        st.dataframe(df)
        df.to_csv("source.csv", index=None)
        st.write("Target Names")
        st.table(data.target_names)
        
  

        




if opt =="Data Analysis":
    gen = st.button("Generate Report")
    down = st.button("Download Report")
    if gen:
        pr = df.head(50).profile_report()
        st_profile_report(pr)
    if down:
        pr.to_file("Analysis.html")


if opt=="Modeling":
    st.session_state['tar']=st.selectbox("Select Target feature:",df.columns)
    
    t_size = st.slider("Training data set size: ", min_value=50,max_value=70,step=10,value=70)
    pre = st.radio("Preprocess the data:",("True","False"))
    
    if st.button("Auto Modeling"):
        
        #st.write(tar)
        setup(df,target=st.session_state.tar,train_size=round(t_size/100,1), preprocess=pre)
        st.info("Model Pipeline")
        set_df = pull()
        st.dataframe(set_df,use_container_width=True)
        best = compare_models()
        
        st.info("Models")
        mod = pull()
        st.dataframe(mod, use_container_width=True)
        #slm = st.selectbox("Choose model to save: ",mod['Model'])
        #if st.button("Save Model"):
        #    m = mod[mod['Model']==slm].index
        #    model=create_model(m) 
        save_model(best,"model")   
        st.info("Model Saved") 

        #chart = st.selectbox("Select Validation report: ",('pipeline','auc','threshold','pr','confusion_matrix','error',
        #'boundary','rfe','learning','manifold','calibration','vc','dimension','feature','feature_all','parameter',
        #'lift','gain','tree','ks'),)
        
        #fig = plot_model(best,plot=chart, display_format='streamlit')
        #st.set_option('deprecation.showPyplotGlobalUse', False)
        #st.write(fig)

if opt == "Predict":
    d=list()
    col = list()
    for x,i in enumerate(df.columns):
        if i==st.session_state.tar:
            break   
        st.text_input(i,key=x)
        col.append(i)
        d.append(st.session_state[x])
    data = pd.DataFrame(columns=col, data=[d])
       #data =pd.DataFrame(data=d,columns=col)
    st.dataframe(data)

    if st.button("Predict"):
        model=load_model('model')
        st.table(predict_model(model, data))
