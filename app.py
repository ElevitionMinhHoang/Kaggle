import streamlit as st
import pandas as pd
import pickle

st.title("Student Academic Warning Predictor")

data = pickle.load(open("model.pkl","rb"))
model = data["model"]
columns = data["columns"]

df = pd.read_csv("train.csv")

student_id = st.text_input("Nhập Student ID")

if st.button("Predict"):

    student = df[df["Student_ID"] == student_id]

    if len(student) == 0:
        st.error("Không tìm thấy sinh viên")

    else:

        X = student.drop(columns=["Student_ID","Academic_Status"], errors="ignore")

        X = pd.get_dummies(X)

        X = X.reindex(columns=columns, fill_value=0)

        pred = model.predict(X)[0]

        if pred == 0:
            st.success("Normal")

        elif pred == 1:
            st.warning("Academic Warning")

        else:
            st.error("Dropout Risk")