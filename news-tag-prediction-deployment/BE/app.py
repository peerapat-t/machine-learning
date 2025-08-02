import pickle
import re
from typing import Dict, List
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pythainlp import thai_characters
from pythainlp.corpus import thai_stopwords
from pythainlp.tokenize import word_tokenize
from pythainlp.util import normalize

# --- Define Constants for Preprocessing ---
URL_PATTERN = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""
SPECIAL_CHARS_PATTERN = r'[!@#$%^&*()\-+=\[\]{};:\'",<.>/?\\|]\n'
WORDS_TO_REMOVE = ['ประชาไท', 'ฯ', '๑', '๒', '๓', '๔', '๕', '๖', '๗', '๘', '๙', '๐']
THAI_STOPWORDS = thai_stopwords()
THAI_CHARACTERS_ONLY = "".join(list(thai_characters))
TARGET_COLUMNS = [
    "politics", "human_rights", "quality_of_life", "foreign_affairs", 
    "society", "environment", "economy", "culture", "labor", 
    "security", "ict", "education"
]

# --- Custom Tokenizer Function (Copied from your script) ---
def custom_thai_tokenizer(text: str) -> List[str]:
    """Processes raw text to a list of clean tokens."""
    text = normalize(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(URL_PATTERN, '', text)
    for word in WORDS_TO_REMOVE:
        text = text.replace(word, '')
    text = re.sub(SPECIAL_CHARS_PATTERN, '', text)
    
    allowed_chars_pattern = f'[^{THAI_CHARACTERS_ONLY}\s]'
    text = re.sub(allowed_chars_pattern, '', text)
    
    tokens = word_tokenize(text, keep_whitespace=False)
    
    filtered_tokens = [token for token in tokens if token not in THAI_STOPWORDS and token.strip() != '']
    
    return filtered_tokens

# --- FastAPI Application Setup ---
app = FastAPI(
    title="Thai News Category Prediction API",
    description="An API to predict the category of a Thai news article.",
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

# --- Load Models at Startup ---
models: Dict[str, object] = {}

@app.on_event("startup")
def load_models():
    """Load all pickled model pipelines into memory when the app starts."""
    print("--- Loading models into memory... ---")
    for column in TARGET_COLUMNS:
        filepath = f'./MODEL/{column}_pipeline.pickle'
        with open(filepath, 'rb') as file:
            models[column] = pickle.load(file)
        print(f"✅ Successfully loaded model for: {column}")
    print("--- Model loading complete. ---")


# --- Define Request and Response Models (using Pydantic) ---
class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    predictions: Dict[str, float]

# --- API Endpoints ---
@app.get("/", tags=["Status"])
def read_root():
    """A simple endpoint to check if the API is running."""
    return {"status": "ok", "message": "Welcome to the Thai News Prediction API!"}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_category(request: TextInput):
    """
    Predicts the probability for each news category given a piece of text.
    
    The tokenizer used for prediction is the same one used for training the models.
    """
    input_text = request.text
    predictions = {}
    
    text_to_predict = [input_text]

    for column, model_pipeline in models.items():
        probability = model_pipeline.predict_proba(text_to_predict)[0][1]
        predictions[column] = round(probability, 4)
        
    return {"predictions": predictions}

# To run the app directly (for development)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)