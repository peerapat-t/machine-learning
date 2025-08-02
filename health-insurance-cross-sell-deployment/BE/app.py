import pickle
import uvicorn
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- 1. APP INITIALIZATION ---
app = FastAPI(
    title="Health Insurance Prediction API",
    description="An API to predict customer interest in Vehicle Insurance based on their health insurance data.",
    version="1.0.0"
)

# Define allowed origins for CORS
origins = ["*"]

# Add CORS middleware to the application
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. LOAD THE PRE-TRAINED PIPELINE ---
with open('./MODEL/logistic_pipeline.pickle', 'rb') as file:
    pipeline = pickle.load(file)


# --- 3. DEFINE THE INPUT DATA MODEL (using Pydantic) ---
class CustomerFeatures(BaseModel):
    Gender: str
    Age: int
    Region_Code: str
    Previously_Insured: int
    Vehicle_Age: str
    Vehicle_Damage: int
    Annual_Premium: float
    Policy_Sales_Channel: str
    Vintage: int

    class Config:
        json_schema_extra = {
            "example": {
                "Gender": "Male",
                "Age": 44,
                "Region_Code": "28.0",
                "Previously_Insured": 0,
                "Vehicle_Age": "> 2 Years",
                "Vehicle_Damage": 1,
                "Annual_Premium": 40454.0,
                "Policy_Sales_Channel": "26.0",
                "Vintage": 217
            }
        }


# --- 4. API ENDPOINTS ---
@app.get("/")
def read_root():
    """
    A simple endpoint to check if the API is running.
    """
    return {"status": "online", "message": "Health Insurance Prediction API is running!"}

@app.post("/predict/")
def predict_interest(features: CustomerFeatures):
    
    input_df = pd.DataFrame([features.dict()])
    
    input_df['Vehicle_Damage'] = input_df['Vehicle_Damage'].map({1: 'Yes', 0: 'No'})
    
    input_df['Region_Code'] = input_df['Region_Code'].astype(str)
    input_df['Policy_Sales_Channel'] = input_df['Policy_Sales_Channel'].astype(str)

    prediction_raw = pipeline.predict(input_df)[0]
    prediction_proba = pipeline.predict_proba(input_df)[0]

    prediction = "Interested" if prediction_raw == 1 else "Not Interested"
    probability = f"{prediction_proba[prediction_raw]:.2%}"

    return {
        "prediction": prediction,
        "probability": probability
    }

# --- 5. RUN THE APP (for local development) ---
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)