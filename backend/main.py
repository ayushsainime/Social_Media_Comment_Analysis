# FastAPI Backend - Phase 4
# Main API application

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
import io
import base64

from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from .models_loader import load_models
from .preprocess import clean_text

# Load models at startup
tfidf, models = load_models()

app = FastAPI(title="Sentiment Analysis API")


# ============= Pydantic Models =============

class ModelName(str, Enum):
    svm = "svm"
    random_forest = "random_forest"
    logistic_regression = "logistic_regression"
    gradient_boosting = "gradient_boosting"
    ada_boost = "ada_boost"
    lgbm = "lgbm"


class PredictionRequest(BaseModel):
    model_name: ModelName = Field(..., description="Select model for prediction")
    text: str = Field(..., description="Text to analyze", min_length=3)


class PredictionResponse(BaseModel):
    model_used: str
    prediction: str
    confidence: Optional[float] = None


class ModelResult(BaseModel):
    """Result from a single model."""
    model_name: str
    prediction: str
    confidence: Optional[float] = None


class PredictAllResponse(BaseModel):
    """Response from all models."""
    text: str
    results: List[ModelResult]


class WordCloudRequest(BaseModel):
    """Request for word cloud generation."""
    text: str = Field(..., description="Text to generate word cloud from", min_length=3)


class WordCloudResponse(BaseModel):
    """Response with base64 encoded word cloud image."""
    image: str  # Base64 encoded PNG image


# ============= API Endpoints =============

@app.get("/")
def home():
    """Root endpoint - API health check."""
    return {"message": "Sentiment Analysis API is running", "status": "ok"}


@app.get("/models")
def available_models_endpoint():
    """Get list of available models."""
    return {"models": list(models.keys())}


def _get_prediction_with_confidence(model, X) -> tuple:
    """
    Get prediction and confidence score from a model.
    
    Returns:
        tuple: (prediction_label, confidence_score)
    """
    sentiment_map = {-1: "negative", 0: "neutral", 1: "positive"}
    
    prediction = model.predict(X)[0]
    sentiment_label = sentiment_map.get(prediction, "unknown")
    
    # Try to get probability/confidence
    confidence = None
    import numpy as np
    if hasattr(model, 'predict_proba'):
        try:
            proba = model.predict_proba(X)[0]
            confidence = float(np.max(proba))
        except Exception:
            pass
    if confidence is None and hasattr(model, 'decision_function'):
        try:
            decision = model.decision_function(X)[0]
            # Handle multi-class (array) vs binary (scalar) decision function output
            if hasattr(decision, '__len__') and not isinstance(decision, (str, bytes)):
                max_decision = float(np.max(decision))
            else:
                max_decision = float(abs(decision))
            # Convert to pseudo-confidence using sigmoid
            confidence = float(1 / (1 + np.exp(-max_decision)))
        except Exception:
            pass

    return sentiment_label, confidence


def _to_confidence_percent(confidence: Optional[float]) -> Optional[float]:
    """Convert 0..1 confidence to percentage, preserving valid 0 values."""
    if confidence is None:
        return None
    try:
        value = float(confidence)
    except Exception:
        return None

    # Keep UI stable even if a model returns a noisy out-of-range score.
    value = max(0.0, min(1.0, value))
    return round(value * 100, 2)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predict sentiment for given text.
    
    Flow:
    1. Clean text
    2. Vectorize using TF-IDF
    3. Predict using selected model
    4. Return sentiment label with confidence
    """
    try:
        # Clean the text
        cleaned_text = clean_text(request.text)
        
        # Vectorize
        X = tfidf.transform([cleaned_text])
        
        # Get model and predict with confidence
        model = models[request.model_name.value]
        sentiment_label, confidence = _get_prediction_with_confidence(model, X)
        
        return PredictionResponse(
            model_used=request.model_name.value,
            prediction=sentiment_label,
            confidence=_to_confidence_percent(confidence)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-all", response_model=PredictAllResponse)
def predict_all(request: PredictionRequest):
    """
    Predict sentiment using ALL available models.
    
    Returns predictions from all models for comparison.
    """
    try:
        # Clean the text
        cleaned_text = clean_text(request.text)
        
        # Vectorize
        X = tfidf.transform([cleaned_text])
        
        # Get predictions from all models
        results = []
        for model_name, model in models.items():
            try:
                sentiment_label, confidence = _get_prediction_with_confidence(model, X)
            except Exception:
                # Keep the comparison response usable even if one model fails.
                sentiment_label, confidence = "unknown", None
            results.append(ModelResult(
                model_name=model_name,
                prediction=sentiment_label,
                confidence=_to_confidence_percent(confidence)
            ))
        
        return PredictAllResponse(
            text=request.text[:100] + "..." if len(request.text) > 100 else request.text,
            results=results
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/wordcloud", response_model=WordCloudResponse)
def generate_wordcloud(request: WordCloudRequest):
    """
    Generate a word cloud from the input text.
    
    Returns a base64 encoded PNG image.
    """
    try:
        # Clean the text
        cleaned_text = clean_text(request.text)
        
        if not cleaned_text.strip():
            raise HTTPException(status_code=400, detail="Text too short after cleaning")
        
        # Generate word cloud
        wc = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(cleaned_text)
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        img_buffer.seek(0)
        
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        return WordCloudResponse(image=img_base64)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
