"""
Configuration centralisée du projet
"""
from pathlib import Path
from typing import Dict, Any

# Chemins du projet
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Créer les dossiers s'ils n'existent pas
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Configuration des données
DATA_CONFIG = {
    "raw_data_path": RAW_DATA_DIR / "creditcard.csv",
    "processed_train_path": PROCESSED_DATA_DIR / "train.csv",
    "processed_test_path": PROCESSED_DATA_DIR / "test.csv",
    "test_size": 0.2,
    "random_state": 42,
    "stratify_col": "Class",
}

# Configuration du modèle
MODEL_CONFIG = {
    "random_state": 42,
    "n_jobs": -1,
    "models": {
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
        },
        "xgboost": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "scale_pos_weight": 1,
        },
        "lightgbm": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "is_unbalance": True,
        },
        "catboost": {
            "iterations": 100,
            "depth": 6,
            "learning_rate": 0.1,
            "auto_class_weights": "Balanced",
        },
    },
}

# Configuration du preprocessing
PREPROCESSING_CONFIG = {
    "scale_features": True,
    "scaler_type": "robust",  # 'standard', 'robust', 'minmax'
    "handle_imbalance": True,
    "imbalance_method": "smote",  # 'smote', 'adasyn', 'borderline_smote'
    "sampling_strategy": 0.5,  # Ratio de la classe minoritaire après resampling
}

# Configuration de l'API
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
    "model_path": MODELS_DIR / "best_model.pkl",
    "scaler_path": MODELS_DIR / "scaler.pkl",
    "threshold": 0.5,  # Seuil de classification
}

# Configuration Streamlit
STREAMLIT_CONFIG = {
    "page_title": "Détection de Fraude - Dashboard",
    "page_icon": "🔐",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# Configuration du monitoring
MONITORING_CONFIG = {
    "drift_threshold": 0.1,
    "alert_email": "moussa.ba.math@gmail.com",
    "check_interval_hours": 24,
}

# Configuration MLflow
MLFLOW_CONFIG = {
    "tracking_uri": "mlruns",
    "experiment_name": "fraud_detection",
    "run_name_prefix": "fraud_model",
}

# Métriques à tracker
METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
    "confusion_matrix",
]

# Features du dataset (après transformation PCA)
FEATURE_NAMES = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]

# Coûts business (en euros)
BUSINESS_COSTS = {
    "false_positive_cost": 25,  # Coût de vérification manuelle
    "false_negative_cost": 500,  # Coût moyen d'une fraude non détectée
    "true_positive_benefit": 450,  # Bénéfice d'une fraude détectée
}

# Configuration des logs
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "INFO",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": LOGS_DIR / "app.log",
            "formatter": "default",
            "level": "DEBUG",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"],
    },
}

