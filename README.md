---
title: SOCIAL MEDIA SENTIMENT ANALYZER
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 3000
pinned: false
---

#  Social Media Sentiment Analyzer

A powerful machine learning web application for analyzing sentiment in social media text. Built with FastAPI backend and Reflex frontend, featuring multiple ML models, word cloud generation, and model comparison capabilities.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.6-green)
![Reflex](https://img.shields.io/badge/Reflex-0.8.28-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

##  Features

- **🔍 Single Model Analysis** - Analyze text sentiment using your preferred ML model
- **Model Comparison** - Compare predictions from all 5 models simultaneously
- **Confidence Scores** - View prediction confidence percentages with visual progress bars
- **Word Cloud Generation** - Generate beautiful word clouds from input text
- **Prediction History** - Track your last 10 predictions in the session
- **Docker Ready** - Fully containerized for easy deployment

---

## Available Models

| Model | Description |
|-------|-------------|
| **SVM** | Support Vector Machine classifier |
| **Random Forest** | Ensemble of decision trees |
| **Logistic Regression** | Probabilistic linear classifier |
| **Gradient Boosting** | Sequential ensemble learning |
| **LightGBM** | Gradient boosting with leaf-wise growth |
| **AdaBoost** | Adaptive boosting classifier |

---

## Sentiment Labels

| Label | Value |
|-------|-------|
| 😊 Positive | `1` |
| 😐 Neutral | `0` |
| 😔 Negative | `-1` |

---

## Project Structure

```
social-media-sentiment-analyzer/
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI application & endpoints
│   ├── models_loader.py     # Model loading utilities
│   └── preprocess.py        # Text preprocessing functions
├──  frontend/
│   ├── __init__.py
│   └── frontend.py          # Reflex frontend application
├──  models/               # Pre-trained ML models (Git LFS)
│   ├── tfidf_vectorizer.pkl
│   ├── svm_model.pkl
│   ├── random_forest_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── gradient_boosting_model.pkl
│   ├── lgbm_model.pkl
│   └── ada_boost_model.pkl
├── .gitattributes           # Git LFS configuration
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
├── requirements.txt
└── start.sh                 # Startup script for Docker
```

---

##  Quick Start

### Prerequisites

- Python 3.11+
- Git LFS (for model files)
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ayushsainime/Multi-Model-Sentiment-Classification-System-.git
   cd Multi-Model-Sentiment-Classification-System-
   ```

2. **Install Git LFS and pull model files**
   ```bash
   git lfs install
   git lfs pull
   ```

3. **Create virtual environment**
   ```bash
   python -m venv .env
   
   # Windows
   .env\Scripts\activate
   
   # Linux/Mac
   source .env/bin/activate
   ```

4. **Install dependencies**
   ```bash
   # IMPORTANT: Install Reflex first (it manages pydantic/sqlmodel compatibility)
   pip install reflex
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

5. **Run the application**
   
   **Option A: Run separately (Development)**
   ```bash
   # Terminal 1 - Start backend (from project root)
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   
   # Terminal 2 - Start Reflex app (from project root)
   # Reflex backend is on 8001 to avoid conflict with FastAPI (8000)
   reflex run --frontend-port 3000 --backend-port 8001
   ```
   
   **Option B: Run with Docker (Production)**
   ```bash
   docker build -t sentiment-analyzer:1.0 .
   docker run --rm -p 8000:8000 -p 8001:8001 -p 3000:3000 sentiment-analyzer:1.0
   ```

6. **Access the application**
   - Frontend: http://localhost:3000
   - API Docs: http://localhost:8000/docs

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API health check |
| `GET` | `/models` | List available models |
| `POST` | `/predict` | Predict sentiment (single model) |
| `POST` | `/predict-all` | Predict sentiment (all models) |
| `POST` | `/wordcloud` | Generate word cloud image |

### Example Requests

**Single Prediction**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "svm", "text": "I love this product!"}'
```

**Response:**
```json
{
  "model_used": "svm",
  "prediction": "positive",
  "confidence": 89.5
}
```

**Compare All Models**
```bash
curl -X POST "http://localhost:8000/predict-all" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "svm", "text": "This movie was okay, nothing special."}'
```

**Response:**
```json
{
  "text": "This movie was okay, nothing special.",
  "results": [
    {"model_name": "svm", "prediction": "neutral", "confidence": 65.2},
    {"model_name": "random_forest", "prediction": "neutral", "confidence": 72.1},
    {"model_name": "logistic_regression", "prediction": "neutral", "confidence": 68.4},
    {"model_name": "gradient_boosting", "prediction": "neutral", "confidence": 70.8},
    {"model_name": "ada_boost", "prediction": "neutral", "confidence": 61.3}
  ]
}
```

---

##  Docker Deployment

```bash
# Build the image
docker build -t sentiment-analyzer:1.0 .

# Run the container
docker run --rm -d \
  --name sentiment-app \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 3000:3000 \
  sentiment-analyzer:1.0

# View logs
docker logs -f sentiment-app
```

---

##  Tech Stack

| Category | Technology |
|----------|------------|
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | Reflex (Python-based React) |
| **ML/DL** | scikit-learn, XGBoost, LightGBM |
| **NLP** | NLTK, spaCy |
| **Visualization** | WordCloud, Matplotlib |
| **Containerization** | Docker |
| **Storage** | Git LFS |

---

##  Model Performance

Models were trained on Twitter and Reddit sentiment datasets. Performance metrics:

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| SVM | ~85% | 0.84 |
| Random Forest | ~83% | 0.82 |
| Logistic Regression | ~82% | 0.81 |
| Gradient Boosting | ~84% | 0.83 |
| AdaBoost | ~81% | 0.80 |

---

##  Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_URL` | `http://localhost:8000` | Backend API URL |

---

##  Text Preprocessing

The preprocessing pipeline includes:
- Lowercase conversion
- URL removal
- Mention (@username) removal
- HTML tag stripping
- Special character removal
- Whitespace normalization

---

##  Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

##  Author

**Ayush Saini**
- GitHub: [@ayushsainime](https://github.com/ayushsainime)
- LinkedIn: [ayush_Saini]()

---

##  Acknowledgments

- Dataset: Twitter and Reddit Sentiment Analysis Dataset
- Inspired by NLP best practices and ML classification techniques

---

⭐ **If you found this project helpful, please give it a star!** ⭐
