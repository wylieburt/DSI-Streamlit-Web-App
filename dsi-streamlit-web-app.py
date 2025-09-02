

# import libraries

import streamlit as st
import pandas as pd
import joblib

# load model pipeline

model = joblib.load("model.joblib")


# Add title and instructions
st.title("Purchase Prediction and Predictions")

st.subheader("Enter customer information and submit for likelihood to purchase")

# Age input form

age = st.number_input(
    label = "01. Enter customer's age",
    min_value = 18,
    max_value = 120,
    value = 35)

# gender input form

gender = st.radio(label = "02. Enter customer's gender",
                  options = ['M', 'F'])


# credit score input form

credit_score = st.slider(
    label = "03. Enter customer's credit score",
    min_value = 0,
    max_value = 1000,
    value = 500)

# submit inputs to model

if st.button("Submit for Prediction"):
    
    # store the input data into a dataframe for prediction
    
    new_data = pd.DataFrame({
        "age": [age], 
        "gender" : [gender], 
        "credit_score" : [credit_score]
    })
    
    # apply model pipeline to the input data and make a prediction
    
    predict_proba = model.predict_proba(new_data)[0][1]

    #output prediction
    
    st.subheader(f"Based on these customer characteristics, our model predicts a purchase probability of {predict_proba:.0%}")
                                       
                                      
    
    
    
    
    
    
    
    
    
    

