import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# Load the trained model and preprocessing artifacts with caching
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model('model.h5')
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, label_encoder_gender, onehot_encoder_geo, scaler

model, label_encoder_gender, onehot_encoder_geo, scaler = load_artifacts()

# Streamlit App UI
st.title('📊 Customer Churn Prediction')
st.markdown('Enter customer demographic and account details to predict churn likelihood.')

col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92, 40)
    tenure = st.slider('Tenure (years)', 0, 10, 3)
    num_of_products = st.slider('Number of Products', 1, 4, 2)

with col2:
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=600)
    balance = st.number_input('Balance ($)', min_value=0.0, value=60000.0, step=1000.0)
    estimated_salary = st.number_input('Estimated Salary ($)', min_value=0.0, value=50000.0, step=1000.0)
    has_cr_card = st.selectbox('Has Credit Card?', [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
    is_active_member = st.selectbox('Is Active Member?', [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform(pd.DataFrame({'Geography': [geography]})).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

st.divider()

# Predict churn
if st.button('Predict Churn', type='primary', use_container_width=True):
    prediction = model.predict(input_data_scaled)
    prediction_proba = float(prediction[0][0])

    st.subheader('Prediction Result')
    st.progress(prediction_proba)
    st.metric(label='Churn Probability', value=f'{prediction_proba * 100:.1f}%')

    if prediction_proba > 0.5:
        st.error('⚠️ **High Risk**: The customer is likely to churn.')
    else:
        st.success('✅ **Low Risk**: The customer is not likely to churn.')

