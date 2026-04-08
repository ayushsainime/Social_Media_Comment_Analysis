# Model Loader - Phase 2
# Load all pre-trained models from the /models folder

import joblib
import os

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def _apply_model_compatibility_fixes(model):
    """Patch older pickled sklearn models for newer runtime versions."""
    class_name = model.__class__.__name__

    # Older AdaBoost pickles may miss this attribute in newer sklearn versions.
    if class_name == "AdaBoostClassifier" and not hasattr(model, "algorithm"):
        model.algorithm = "SAMME"

    return model


def load_models():
    """
    Load all models and TF-IDF vectorizer from the models folder.
    
    Returns:
        tuple: (tfidf_vectorizer, models_dict)
    """
    # Load TF-IDF vectorizer
    tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    tfidf = joblib.load(tfidf_path)
    
    # Load all classifier models
    models = {
        "svm": _apply_model_compatibility_fixes(joblib.load(os.path.join(MODELS_DIR, "svm_model.pkl"))),
        "random_forest": _apply_model_compatibility_fixes(joblib.load(os.path.join(MODELS_DIR, "random_forest_model.pkl"))),
        "logistic_regression": _apply_model_compatibility_fixes(joblib.load(os.path.join(MODELS_DIR, "logistic_regression_model.pkl"))),
        "gradient_boosting": _apply_model_compatibility_fixes(joblib.load(os.path.join(MODELS_DIR, "gradient_boosting_model.pkl"))),
        "ada_boost": _apply_model_compatibility_fixes(joblib.load(os.path.join(MODELS_DIR, "ada_boost_model.pkl"))),
        "lgbm": _apply_model_compatibility_fixes(joblib.load(os.path.join(MODELS_DIR, "lgbm_model.pkl"))),
    }
    
    return tfidf, models


def get_available_models():
    """
    Get list of available model names.
    
    Returns:
        list: List of model names
    """
    _, models = load_models()
    return list(models.keys())


if __name__ == "__main__":
    # Test the loader
    tfidf, models = load_models()
    print(f"Loaded TF-IDF vectorizer")
    print(f"Available models: {list(models.keys())}")
