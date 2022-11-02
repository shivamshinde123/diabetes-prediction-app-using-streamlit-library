import streamlit as st
import pandas as pd


def app():

    df = pd.read_csv('diabetes.csv')
    columns = df.columns
    
    st.title('Dataset Feature Distribution')

    for column in columns.drop('Outcome'):
        st.area_chart(df[column])

    
app()
    