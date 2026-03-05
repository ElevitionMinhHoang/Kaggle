import streamlit as st
import pandas as pd
import pickle

st.title("Student Academic Warning Predictor")

# load model
data = pickle.load(open("model.pkl","rb"))
model = data["model"]
columns = data["columns"]

# load dataset
df = pd.read_csv("train.csv")

name = st.text_input("Nhập tên sinh viên")

if st.button("Predict"):

    if name == "":
        st.warning("Nhập tên sinh viên")
    
    else:
        student = df[df["Student_Name"] == name]

        if len(student) == 0:
            st.error("Không tìm thấy sinh viên")

        else:
            X = student.drop(columns=["Student_Name","Academic_Status"], errors="ignore")

            X = pd.get_dummies(X)

            # đảm bảo đúng feature
            X = X.reindex(columns=columns, fill_value=0)

            pred = model.predict(X)[0]

            if pred == 0:
                st.success("Normal - Sinh viên học bình thường")

            elif pred == 1:
                st.warning("Academic Warning")

            else:
                st.error("Dropout Risk")