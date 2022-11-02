import streamlit as st

def app():

    st.title('Metadata')

    st.write(f"Data source: https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset")

    st.write("Data Background:")

    st.code('''This dataset is originally from the National Institute of Diabetes and Digestive and Kidney
Diseases. The objective of the dataset is to diagnostically predict whether a patient has diabetes,
based on certain diagnostic measurements included in the dataset. Several constraints were placed
on the selection of these instances from a larger database. In particular, all patients here are females
at least 21 years old of Pima Indian heritage.''')

    st.code('''Information about dataset attributes -

                Pregnancies: To express the Number of pregnancies

                Glucose: To express the Glucose level in blood

                BloodPressure: To express the Blood pressure measurement

                SkinThickness: To express the thickness of the skin

                Insulin: To express the Insulin level in blood

                BMI: To express the Body mass index

                DiabetesPedigreeFunction: To express the Diabetes percentage

                Age: To express the age

                Outcome: To express the final result 1 is Yes and 0 is No''')


app()