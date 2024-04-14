# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 21:35:33 2024

@author: HP
"""

import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

#st.set_page_config(layout = "wide")

scaler = joblib.load('D:/Machine learning Projects/Restraunt_rating/Scaler.pkl')

st.title("Restaurant Rating Prediction System")


st.caption("This helps you to predict a restaurant review class")

st.divider()

avg_cost = st.number_input("Please enter the estimated cost for two",min_value=50, max_value=99999,value=1000,step=200)
tabel_booking = st.selectbox("Restaurant has table booking?",["Yes","No"])
online_delivery = st.selectbox("Restaurant has online booking?",["Yes","No"])
price_range = st.selectbox("What is the price range(1-chepeast, 4-most expensive)",[1,2,3,4])

predictbutton = st.button("Predict the review!")

st.divider()

saved_model= joblib.load('D:/Machine learning Projects/Restraunt_rating/mlmodel.pkl')
print(type(saved_model))


booking_status = 1 if tabel_booking == "Yes" else 0
delivery_status = 1 if online_delivery == "Yes" else 0
    
values = [[avg_cost,booking_status,delivery_status,price_range]]
my_X_values = np.array(values)

X = scaler.transform(my_X_values)



if predictbutton:
    st.snow()
    
    prediction = saved_model.predict(X)
    
    if prediction < 2.5:
        st.write("Poor")
    elif prediction < 3.5:
        st.write("Average")
    elif prediction < 4.0:
        st.write("Good")
    elif prediction < 4.5:
        st.write("Very Good")
    else:
        st.wrie("Excellent")
