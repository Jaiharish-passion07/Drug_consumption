import streamlit as st
import preprocess
import predict_result
from PIL import Image
import time

st.title('Drug Consumption')
sidebar_=st.sidebar
sidebar_.title("Choose Model")
model_opt=sidebar_.selectbox("Simple",('None','Decision Tree','Logistic Regression','Support Vector Machine',
                                  'Naive Bayes','Perceptron'),label_visibility='hidden')

placeholder = st.empty()

# Replace the chart with several elements:
if model_opt == 'None':

    with placeholder.container():

        image = Image.open('drug.jpg')

        st.image(image, caption='')
        st.write("Asked if Thursday's arrest of heroin and ganja peddlers had any "
                 "connection with the recently arrested international drug trafficker Tonys gang, Mr.Anand said every gang was interlinked with another, since in many cases, the quality of contraband was similar."
                 "Further sharing the progress in the Pudding and Mink pub case, he said that they are likely to arrest four or five customers, who were prospecti")
        st.write("The consumption of drugs and narcotic substances like heroin, cannabis and other opioids has increased significantly in India, one of India’s top narcotics cops has claimed. Sanjay Kumar Singh, Deputy Director-General (Operations) at "
                 "Narcotics Control Bureau (NCB), said drug abuse among the youth has also increased.")
if model_opt!='None':
    placeholder.empty()
    st.subheader("Select all Attributes to predict the Data")

    col1, col2= st.columns(2)
    with col1:
        age = st.selectbox('Age', ('18-24', '25-34', '35-44', '45-54', '55-64', '65+'))

        gender = st.radio(
            "Gender",
            ('Male', 'Female'))

        if gender == 'Male':
            pass
        else:
            pass

        education = st.selectbox("Education", (
        'Doct Deg', 'Mast Deg', 'Prof Cert', 'Univ Deg', 'Some Clg', 'LS@18Y', 'LS@17Y', 'LS@16Y', 'LSB 16Y'))

        country = st.selectbox("Country",
                                   ('UK', 'Canada', 'USA', 'Other', 'Australia', 'Reb of Inreland', 'New Zealand'))

        ethinicty = st.selectbox("Ethinicty", (
        'Mixed-White/Asian', 'White', 'other', 'Mixed-White/Black', 'Asian', 'Black', 'Mixed-Black/Asian'))

        impulsive_val = st.slider(
                'Impulsive values',
                -2.5, 2.9, 0.0)


    with col2:
        n_score_values = st.slider(
            'Nscore',
            -3.5, 3.3, 0.0)

        e_score_values = st.slider(
            'Escore',
            -3.2, 3.2, 0.0)

        o_score_values = st.slider(
            'Oscore',
            -3.5, 2.9, 0.0)

        c_score_values = st.slider(
            'Cscore',
            -3.4, 3.4, 0.0)

        a_score_values = st.slider(
            'Ascore',
            -3.4, 3.4, 0.0)

        ss_val = st.slider(
            'SS Values',
            -2.07, 1.94, 0.0)

    inpt_features=preprocess.encdoding_data(age,education,n_score_values,e_score_values,o_score_values,a_score_values,
                              c_score_values,impulsive_val,ss_val,country,ethinicty,gender)
    # st.write(inpt_features)
    predict=st.button("Predict")
    if predict:
        X=inpt_features
        if model_opt=='Decision Tree':
            model_file="ml_saved_models/dec_tree.pkl"
            result=predict_result.model_call(model_file,X)
            with st.spinner('Wait for it...'):
                time.sleep(3)
            st.success(f"The predicted result is {predict_result.fetch_result(result[0])}", icon="✅")

        elif model_opt=='Logistic Regression':
            model_file="ml_saved_models/log_reg.pkl"
            result=predict_result.model_call(model_file,X)
            with st.spinner('Wait for it...'):
                time.sleep(3)
            st.success(f"The predicted result is {predict_result.fetch_result(result[0])}", icon="✅")

        elif model_opt == 'Support Vector Machine':
            model_file="ml_saved_models/SVM.pkl"
            result=predict_result.model_call(model_file,X)
            with st.spinner('Wait for it...'):
                time.sleep(3)
            st.success(f"The predicted result is {predict_result.fetch_result(result[0])}", icon="✅")

        elif model_opt == 'Naive Bayes':
            model_file="ml_saved_models/naiva_bayes.pkl"
            result=predict_result.model_call(model_file,X)
            with st.spinner('Wait for it...'):
                time.sleep(3)
            st.success(f"The predicted result is {predict_result.fetch_result(result[0])}", icon="✅")

        elif model_opt == 'Perceptron':
            model_file="ml_saved_models/perceptron.pkl"
            result=predict_result.model_call(model_file,X)
            with st.spinner('Wait for it...'):
                time.sleep(3)
            st.success(f"The predicted result is {predict_result.fetch_result(result[0])}", icon="✅")
    else:
        pass
