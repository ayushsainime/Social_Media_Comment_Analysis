from fastapi import FastAPI , HTTPException
from pydantic import BaseModel , Field ,field_validator
import joblib
import numpy as np

app = FastAPI(title="twitter-reddit text classification  API")
import os 

# =========================
# COMPLETE NLP CLEANING PIPELINE (spaCy)
# =========================


# Load spaCy model (run once)
# python -m spacy download en_core_web_sm
import re,string
import spacy
import contractions

nlp = spacy.load("en_core_web_sm")

CUSTOM_STOPWORDS = nlp.Defaults.stop_words - {"not", "no", "but", "however"}


def clean_text(text):
    text = str(text).lower()

    """
    Cleans raw text using regex + spaCy:
    - Replace URLs with 'URL'
    - Replace mentions with 'USER'
    - Replace <3 with 'HEART'
    - Remove HTML tags
    - Remove stopwords
    - Remove punctuation
    - Remove numbers
    - Remove alphanumeric tokens (abc123)
    - Lemmatize words
    - Lowercase output
    """

    #---------------contractions----------------
    text = contractions.fix( text)  
    


    # -------- REGEX CLEANING --------
    text = re.sub(r'https?:\/\/\S*|www\.\S+', 'URL', text)
    text = re.sub(r'@\S+', 'USER', text)
    text = re.sub(r'<3', 'HEART', text)
    text = re.sub(r'<.*?>', '', text)   # remove HTML

    # -------- SPACY PIPELINE --------
    doc = nlp(text)

    tokens = [
        token.lemma_.lower()
        for token in doc
        if token.text.lower() not in CUSTOM_STOPWORDS   # stopwords
        and not token.is_punct        # punctuation
        and not token.like_num        # numbers
        and token.is_alpha            # removes alphanumeric words
    ]

    return " ".join(tokens)






'''
tfidf = joblib.load(r"E:\twitter-and-reddit-sentimental-analysis\models\tfidf_vectorizer.pkl") 

# Load model at startup
models = {
    "svm": joblib.load(r"E:\twitter-and-reddit-sentimental-analysis\models\svm_model.pkl"),
    "random_forest": joblib.load(r"E:\twitter-and-reddit-sentimental-analysis\models\random_forest_model.pkl"),
    "logistic_regression": joblib.load(r"E:\twitter-and-reddit-sentimental-analysis\models\logistic_regression_model.pkl"),
    "gradient_boosting": joblib.load(r"E:\twitter-and-reddit-sentimental-analysis\models\gradient_boosting_model.pkl"),
    "lgbm": joblib.load(r"E:\twitter-and-reddit-sentimental-analysis\models\lgbm_model.pkl") , 
    "ada_boost": joblib.load(r"E:\twitter-and-reddit-sentimental-analysis\models\ada_boost_model.pkl")  , 

}'''


# 1. Get the directory where this current script (fast_api.py) is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Define the path to the 'models' folder relative to this script
MODELS_DIR = os.path.join(BASE_DIR, "models")

# 3. Load using the joined paths
tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))

models = {
    "svm": joblib.load(os.path.join(MODELS_DIR, "svm_model.pkl")),
    "random_forest": joblib.load(os.path.join(MODELS_DIR, "random_forest_model.pkl")),
    "logistic_regression": joblib.load(os.path.join(MODELS_DIR, "logistic_regression_model.pkl")),
    "gradient_boosting": joblib.load(os.path.join(MODELS_DIR, "gradient_boosting_model.pkl")),
    "ada_boost": joblib.load(os.path.join(MODELS_DIR, "ada_boost_model.pkl"))
}

#jojo = joblib.load(r"E:\twitter-and-reddit-sentimental-analysis\models\svm_model.pkl")


'''
kakaji = "i will spread peace and kindness  all this world needs is love  " 
#print( text1) 
text1 = clean_text(kakaji) 
print( text1) 

text1 = tfidf.transform([kakaji])
result = models["svm"].predict(text1)[0]
print(result)

'''



from enum import Enum
class ModelName(str, Enum):
    svm = "svm"
    random_forest = "random_forest"
    logistic_regression = "logistic_regression"
    gradient_boosting = "gradient_boosting"
    ada_boost = "ada_boost"


class PredictionRequest( BaseModel ) : 
    model_name : ModelName = Field(..., description = "Select from: naive_bayes, logistic_regression, svm, lgbm, random_forest, xgboost" ,
                                  example="choose any one from  logistic_regression, svm , random_forest, xgboost") 
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str):
        if len(value.strip()) < 3:
            raise ValueError("Text must contain at least 3 characters")
        return value



@app.get("/")
def home():
    return {"message": "twitter-reddit sentiment classification API is running"}



@app.post('/predict')
def predict(request: PredictionRequest):
    try:
        X = clean_text(request.text)
        X = tfidf.transform([request.text])
        
        model = models[request.model_name.value]
        prediction = model.predict(X)[0]

        if prediction == -1:
            prediction = "negative"
        elif prediction == 1:   
            prediction = "positive"
        else : 
            prediction = "neutral"    

        return {
            "model_used": request.model_name.value,
            "prediction": prediction
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/models")
def available_models():
    return {"models": list(models.keys())}







