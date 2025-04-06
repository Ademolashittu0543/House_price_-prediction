import streamlit as st
import pandas as pd
import pickle

# Load the trained model
try:
    model = pickle.load(open('modell.pkl', 'rb'))
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.title("🏡 House Price Prediction App")
st.write("Fill in the details below to estimate the house price.")

# Input fields
bedrooms = st.slider("Number of Bedrooms", 0, 10, 3)
bathrooms = st.slider("Number of Bathrooms", 0.0, 5.0, 2.0, step=0.25)
sqft_living = st.number_input("Living Area (sqft)", min_value=100, value=1500)
sqft_lot = st.number_input("Lot Size (sqft)", min_value=500, value=3000)
floors = st.slider("Number of Floors", 1, 3, 1)
waterfront = st.selectbox("Waterfront View?", [0, 1])
view = st.slider("View Rating", 0, 4, 0)
condition = st.slider("Condition Rating", 1, 5, 3)
grade = st.slider("Grade Rating", 1, 13, 7)
sqft_above = st.number_input("Square Feet Above", min_value=100, value=1200)
sqft_basement = st.number_input("Square Feet Basement", min_value=0, value=300)
house_age = st.slider("House Age", 0, 120, 20)
city = st.selectbox("City", [
    'Shoreline', 'Seattle', 'Kent', 'Bellevue', 'Redmond', 'Maple Valley',
    'North Bend', 'Lake Forest Park', 'Sammamish', 'Auburn', 'Des Moines',
    'Bothell', 'Federal Way', 'Kirkland', 'Issaquah', 'Woodinville',
    'Normandy Park', 'Fall City', 'Renton', 'Carnation', 'Snoqualmie',
    'Duvall', 'Burien', 'Covington', 'Inglewood-Finn Hill', 'Kenmore',
    'Newcastle', 'Mercer Island', 'Black Diamond', 'Ravensdale', 'Clyde Hill',
    'Algona', 'Skykomish', 'Tukwila', 'Vashon', 'Yarrow Point', 'SeaTac',
    'Medina', 'Enumclaw', 'Snoqualmie Pass', 'Pacific', 'Beaux Arts Village',
    'Preston', 'Milton'
])

# Combine features (including raw 'city' column)
input_data = pd.DataFrame([{
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'sqft_living': sqft_living,
    'sqft_lot': sqft_lot,
    'floors': floors,
    'waterfront': waterfront,
    'view': view,
    'condition': condition,
    'grade': grade,
    'sqft_above': sqft_above,
    'sqft_basement': sqft_basement,
    'house_age': house_age,
    'city': city  # pass city as-is, do not one-hot encode
}])

# Prediction
if st.button("Predict Price"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"Estimated House Price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
