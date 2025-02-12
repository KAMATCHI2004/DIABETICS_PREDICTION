import streamlit as st
import pickle

a = ['1','0']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit_transform(a)

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

P = st.text_input("Enter Pregnancies value:")
G = st.text_input("Enter Glucose value:")
BP = st.text_input("Enter BloodPressure value:")
SK = st.text_input("Enter SkinThickness value:")
I = st.text_input("Enter Insulin value:")
B = st.text_input("Enter BMI value:")
DPF = st.text_input("Enter Diabetes Pedigree Function Value:")
A = st.text_input("Enter Age:")


if st.button("Submit"):
    try:
        # Convert inputs to numeric values
        data = [[float(P), float(G), float(BP), float(SK), float(I), float(B), float(DPF), float(A)]]
        pred = model.predict(data)
        st.write(f"Prediction: {pred[0]}")
    except ValueError:
        st.write("Please enter valid numeric values for all inputs.")
