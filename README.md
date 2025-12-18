# Multi Model Sentiment Classification System

An end to end NLP application for classifying social media text using multiple machine learning models, exposed through a FastAPI backend and an interactive Streamlit frontend, fully containerized using Docker.

---

## 📌 Project Overview

This project allows users to analyze text from platforms such as Twitter and Reddit and generate sentiment predictions in real time.  
Users can select from multiple trained classical machine learning models and compare predictions using the same input text.

The system is designed to be fully reproducible and portable using Docker, so it can be run on any machine without manual dependency setup.

---

## 🧠 Key Features

* Real time text classification using TF IDF based feature extraction  
* Multiple ML models including SVM, Logistic Regression, Random Forest, Gradient Boosting, AdaBoost, and LightGBM  
* Model selection at inference time  
* FastAPI backend for scalable inference  
* Streamlit frontend for interactive usage  
* Dockerized deployment for portability and reproducibility  

---

## 🛠️ Technologies Used

* Python  
* FastAPI  
* Streamlit  
* Scikit learn  
* LightGBM  
* spaCy  
* TF IDF Vectorization  
* Docker  

---

## 🏗️ System Architecture

* Streamlit frontend for user interaction  
* FastAPI backend for model inference  
* TF IDF vectorizer for text feature extraction  
* Multiple trained ML models loaded at startup  
* Single Docker container running both frontend and backend  

---

## 🚀 Running the Project Using Docker (Recommended)

This is the easiest way to run the project.  
No Python installation or environment setup is required.

### Prerequisites

* Docker installed and running on your system  

---

### Step 1 Pull the Docker Image
step 1 

OPEN THE TERMINAL IN DOCKER AND PASTE-
```bash
docker pull ayushsainime/twitter-reddit-nlp:1.0
```

STEP 2 
RUN THE IMAGE : 
```
docker run -p 8000:8000 -p 8501:8501 ayushsainime/twitter-reddit-nlp:1.0
```

Step 3 Open in Browser

Streamlit frontend

```
http://localhost:8501
```
FastAPI documentation

```
http://localhost:8000/docs
```
---

### 🧪 How to Use the Application

1 Open the Streamlit interface in your browser
2 Select a machine learning model
3 Enter the text you want to analyze
4 Click Predict
5 View the predicted sentiment 


### 🧑‍💻 Running from Source Code (Without Docker)

Step 1 Clone the Repository
```bash
git clone https://github.com/ayushsainime/Multi-Model-Sentiment-Classification-System-.git
cd Multi-Model-Sentiment-Classification-System-
```
Step 2 Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
```

Step 3 Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Step 4 Run FastAPI Backend
```bash
uvicorn fast_api:app --host 0.0.0.0 --port 8000
```

Step 5 Run Streamlit Frontend
```bash
streamlit run frontend_app.py
```









