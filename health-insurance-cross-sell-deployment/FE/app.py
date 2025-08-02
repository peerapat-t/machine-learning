import streamlit as st
import requests
import json

# --- 1. APP TITLE AND DESCRIPTION ---
st.set_page_config(page_title="Car Insurance Signal", page_icon="🚗")
st.title('🚗 Car Insurance Selling Signal')

# --- 2. BACKEND API URL ---
API_URL = "http://127.0.0.1:8000/predict/"

# --- 3. DEFINE LISTS FOR SELECTBOXES ---
region_code_list = [str(i) for i in range(53)] 
policy_sales_channel_list = ['1', '2', '3', '4', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '73', '74', '75', '76', '78', '79', '80', '81', '82', '83', '84', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '163']


# --- 4. PREDICTION FORM ---
with st.form("prediction_form"):
    st.header("Enter Customer Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ("Male", "Female"))
        age = st.slider("Age", min_value=18, max_value=100, value=40)
        previously_insured = st.selectbox("Previously Insured?", (0, 1), format_func=lambda x: "Yes" if x == 1 else "No")
        vehicle_damage = st.selectbox("Vehicle has Damage?", (1, 0), format_func=lambda x: "Yes" if x == 1 else "No")
        
    with col2:
        region_code = st.selectbox("Region Code", region_code_list, index=28)
        vehicle_age = st.selectbox("Vehicle Age", ("< 1 Year", "1-2 Year", "> 2 Years"))
        annual_premium = st.number_input("Annual Premium (€)", min_value=2000.0, value=30000.0)
        vintage = st.slider("Days with Company (Vintage)", min_value=10, max_value=300, value=150)
    
    policy_sales_channel = st.selectbox("Policy Sales Channel", policy_sales_channel_list, index=24) # Default to 26.0

    submitted = st.form_submit_button("Predict")

# --- 5. PREDICTION LOGIC ---
if submitted:
    payload = {
        "Gender": gender,
        "Age": age,
        "Region_Code": region_code,
        "Previously_Insured": previously_insured,
        "Vehicle_Age": vehicle_age,
        "Vehicle_Damage": vehicle_damage,
        "Annual_Premium": annual_premium,
        "Policy_Sales_Channel": policy_sales_channel,
        "Vintage": vintage
    }

    with st.spinner("Getting prediction..."):
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()

            result = response.json()

            st.subheader("Prediction Result")
            
            prediction = result.get("prediction")
            probability = result.get("probability")

            if prediction == "Interested":
                st.success(f"The customer is LIKELY INTERESTED in Vehicle Insurance.")
            else:
                st.warning(f"The customer is LIKELY NOT INTERESTED in Vehicle Insurance.")
            
            if probability:
                st.info(f"Confidence: {probability}")

        except requests.exceptions.RequestException as e:
            st.error(f"Error: Could not connect to the API. Please ensure the backend server is running. Details: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")