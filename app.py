import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Input Fields
city_encoded = st.number_input("City (Encoded)", min_value=0, step=1)
house_age = st.number_input("House Age (years)", min_value=0, step=1)
sqft_living = st.number_input("Living Area (sqft)", min_value=100, step=10)
sqft_lot = st.number_input("Lot Area (sqft)", min_value=500, step=10)
bedrooms = st.number_input("Bedrooms", min_value=1, step=1)
bathrooms = st.number_input("Bathrooms", min_value=1.0, step=0.5)
floors = st.number_input("Floors", min_value=1, step=1)
waterfront = st.selectbox("Waterfront View", [0, 1])
view = st.number_input("View Rating", min_value=0, step=1)
condition = st.number_input("Condition Rating", min_value=1, step=1)
sqft_basement = st.number_input("Basement Area (sqft)", min_value=0, step=10)


# Prediction
if st.button("Predict Price"):
    # Convert inputs into a NumPy array
    input_features = np.array([[city_encoded, house_age, sqft_living, sqft_lot, bedrooms,
                                bathrooms, floors, waterfront, view, condition,
                                sqft_basement]])
    
    # Make prediction
    predicted_price = model.predict(input_features)[0]

    # Display Result
    st.success(f"🏠 Estimated House Price: **${predicted_price:,.2f}**")