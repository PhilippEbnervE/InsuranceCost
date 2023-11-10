import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Create Layout of Streamlit Page
st.header("Philipp Models for the Individual Assignment")

# Load the saved model
pickle_in = open('model_insurance_final.pkl', 'rb') 
classifier = pickle.load(pickle_in)

scaler_file_path = 'scaler_final.pkl'  
with open(scaler_file_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define a function to make predictions
def predict_cost(age, sex, bmi, children, smoker):
    
    # Create a DataFrame with user inputs
    user_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
    })
    
    # Scale the input data using the saved scaler
    data_scaled = scaler.transform(user_data)

    # Make the prediction
    prediction = classifier.predict(data_scaled)

    return prediction[0]

# Streamlit UI
st.title('Insurance Cost Prediction App')
st.write('Enter the following information to predict the insurance cost.')

# User inputs
age = st.number_input('What is your age?', min_value=0)
sex = st.radio('What is your sex?', ['Male', 'Female'])
bmi = st.number_input('Enter your Body Mass Index', min_value=0.0)
children = st.number_input('Enter how many children you have', min_value=0)
smoker = st.radio('Do you smoke?', ['Yes', 'No'])

# Convert categorical values to numerical
sex = 1 if sex == 'Male' else 0
smoker = 1 if smoker == 'Yes' else 0

# Make the prediction
if st.button('Predict Cost'):
    prediction = round(predict_cost(age, sex, bmi, children, smoker),2)
    st.write(f'Predicted Insurance Cost: $ {prediction}')