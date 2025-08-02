from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import List
from functions import get_recommendations

app = FastAPI(title="Market Basket Recommendation API")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Market Basket Recommendation API is running."}

class Transaction(BaseModel):
    items: List[str]

@app.post("/predict/")
def recommend_items(txn: Transaction):
    try:
        recommendations = get_recommendations(set(txn.items))
        return {"input_items": txn.items, "recommended_items": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)