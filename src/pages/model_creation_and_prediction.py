import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np

def app():

    df = pd.read_csv('diabetes.csv')
    X = df.drop('Outcome',axis=1)
    y = df['Outcome']

    rfc = RandomForestClassifier(max_depth=2,random_state=2346)
    model = rfc.fit(X,y)

    with open('model.pkl','wb') as file:
        pickle.dump(model, file)

    
    feature_min = []
    feature_max = []
    feature_mean = []
    input_from_user = []

    for feature in df.columns:
        feature_min.append(min(df[feature]))
        feature_max.append(max(df[feature]))
        feature_mean.append(np.mean(df[feature]))

    for index, feature in enumerate(df.columns.drop('Outcome')):
        # st.write(feature)
        user_input = st.slider(feature,float(feature_min[index]),float(feature_max[index]),float(feature_mean[index]),0.1)
        input_from_user.append(user_input)

    st.markdown('***0 represents the people predicted with no diabetes***')
    st.markdown('***1 represents the people predicted with diabetes***')

    if st.button('Predict Diabetes'):
        input = np.array(input_from_user).reshape(1,-1)
        prediction = model.predict(input)
        proba_pred = model.predict_proba(input)
        
        st.header('Predictions:')
        st.write("Outcome of prediction")
        st.write(prediction)
        st.write("Probability of each class")
        st.write(proba_pred)
    

st.title('Diabetes Prediction')
st.markdown("**Please adjust the slider according to your parameters and then click on the predict button to get the prediction**")
st.markdown("**Inputs:**")
app()

    

    


    

