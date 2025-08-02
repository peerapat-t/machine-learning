import streamlit as st
import pandas as pd
import requests

# ✅ MUST be the first Streamlit command
st.set_page_config(page_title="🛍️ Market Basket Recommender", layout="centered")


API_URL = "http://localhost:8000/predict/"  # FastAPI backend URL

# Load item list from Excel
@st.cache_data
def load_item_names():
    df_items = pd.read_excel("item_name_final.xlsx")
    return sorted(df_items['Description'].dropna().unique().tolist())

# Load items
item_names = load_item_names()

# Streamlit UI
st.title("🛍️ Market Basket Recommendation System")
st.markdown("Select purchased items from the list below to get recommendations:")

selected_items = st.multiselect("Select Purchased Items", item_names)

if st.button("Get Recommendations"):
    if not selected_items:
        st.warning("Please select at least one item.")
    else:
        with st.spinner("Fetching recommendations..."):
            try:
                response = requests.post(API_URL, json={"items": selected_items})
                response.raise_for_status()
                result = response.json()

                st.success("Recommended Items:")
                if result["recommended_items"]:
                    for rec in result["recommended_items"]:
                        st.markdown(f"- {rec}")
                else:
                    st.info("No recommendations found for selected items.")
            except Exception as e:
                st.error(f"Failed to get recommendation: {e}")
