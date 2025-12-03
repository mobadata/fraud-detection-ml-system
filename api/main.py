"""
API FastAPI pour la détection de fraude

Endpoints:
- GET / : Health check
- POST /predict : Prédiction unique
- POST /predict_batch : Prédictions en batch
- GET /model_info : Informations sur le modèle
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Ajouter le dossier src au path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# App FastAPI
app = FastAPI(
    title="Fraud Detection API",
    description="API de détection de fraude sur transactions bancaires",
    version="1.0.0"
)

# Charger le modèle et le scaler au démarrage
MODEL_PATH = Path(__file__).parent.parent / "models" / "best_model_random_forest.pkl"
SCALER_PATH = Path(__file__).parent.parent / "models" / "scaler.pkl"

model = None
scaler = None

# Feature names (30 features: V1-V28 + Time + Amount)
FEATURE_NAMES = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']


@app.on_event("startup")
async def load_model():
    """Charge le modèle et le scaler au démarrage"""
    global model, scaler
    
    try:
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            print(f"✅ Modèle chargé : {MODEL_PATH}")
        else:
            print(f"⚠️ Modèle non trouvé à {MODEL_PATH}")
            print("   L'API fonctionnera mais /predict ne sera pas disponible")
            
        if SCALER_PATH.exists():
            scaler = joblib.load(SCALER_PATH)
            print(f"✅ Scaler chargé : {SCALER_PATH}")
        else:
            print(f"⚠️ Scaler non trouvé à {SCALER_PATH}")
    except Exception as e:
        print(f"❌ Erreur au chargement : {e}")


# Modèles Pydantic pour la validation
class Transaction(BaseModel):
    """Une transaction unique"""
    V1: float = Field(..., description="Feature V1 (PCA component)")
    V2: float = Field(..., description="Feature V2 (PCA component)")
    V3: float = Field(..., description="Feature V3 (PCA component)")
    V4: float = Field(..., description="Feature V4 (PCA component)")
    V5: float = Field(..., description="Feature V5 (PCA component)")
    V6: float = Field(..., description="Feature V6 (PCA component)")
    V7: float = Field(..., description="Feature V7 (PCA component)")
    V8: float = Field(..., description="Feature V8 (PCA component)")
    V9: float = Field(..., description="Feature V9 (PCA component)")
    V10: float = Field(..., description="Feature V10 (PCA component)")
    V11: float = Field(..., description="Feature V11 (PCA component)")
    V12: float = Field(..., description="Feature V12 (PCA component)")
    V13: float = Field(..., description="Feature V13 (PCA component)")
    V14: float = Field(..., description="Feature V14 (PCA component)")
    V15: float = Field(..., description="Feature V15 (PCA component)")
    V16: float = Field(..., description="Feature V16 (PCA component)")
    V17: float = Field(..., description="Feature V17 (PCA component)")
    V18: float = Field(..., description="Feature V18 (PCA component)")
    V19: float = Field(..., description="Feature V19 (PCA component)")
    V20: float = Field(..., description="Feature V20 (PCA component)")
    V21: float = Field(..., description="Feature V21 (PCA component)")
    V22: float = Field(..., description="Feature V22 (PCA component)")
    V23: float = Field(..., description="Feature V23 (PCA component)")
    V24: float = Field(..., description="Feature V24 (PCA component)")
    V25: float = Field(..., description="Feature V25 (PCA component)")
    V26: float = Field(..., description="Feature V26 (PCA component)")
    V27: float = Field(..., description="Feature V27 (PCA component)")
    V28: float = Field(..., description="Feature V28 (PCA component)")
    Time: float = Field(..., description="Temps en secondes depuis la première transaction")
    Amount: float = Field(..., ge=0, description="Montant de la transaction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "V1": -1.3598071336738,
                "V2": -0.0727811733098497,
                "V3": 2.53634673796914,
                "V4": 1.37815522427443,
                "V5": -0.338320769942518,
                "V6": 0.462387777762292,
                "V7": 0.239598554061257,
                "V8": 0.0986979012610507,
                "V9": 0.363786969611213,
                "V10": 0.0907941719789316,
                "V11": -0.551599533260813,
                "V12": -0.617800855762348,
                "V13": -0.991389847235408,
                "V14": -0.311169353699879,
                "V15": 1.46817697209427,
                "V16": -0.470400525259478,
                "V17": 0.207971241929242,
                "V18": 0.0257905801985591,
                "V19": 0.403992960255733,
                "V20": 0.251412098239705,
                "V21": -0.018306777944153,
                "V22": 0.277837575558899,
                "V23": -0.110473910188767,
                "V24": 0.0669280749146731,
                "V25": 0.128539358273528,
                "V26": -0.189114843888824,
                "V27": 0.133558376740387,
                "V28": -0.0210530534538215,
                "Time": 406.0,
                "Amount": 149.62
            }
        }


class TransactionBatch(BaseModel):
    """Batch de transactions"""
    transactions: List[Transaction]


class PredictionResponse(BaseModel):
    """Réponse de prédiction"""
    is_fraud: bool
    fraud_probability: float
    confidence: str
    recommendation: str


class BatchPredictionResponse(BaseModel):
    """Réponse pour un batch de prédictions"""
    predictions: List[PredictionResponse]
    total: int
    frauds_detected: int
    fraud_rate: float


# Endpoints

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "online",
        "message": "Fraud Detection API is running",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }


@app.get("/model_info")
async def model_info():
    """Informations sur le modèle"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    return {
        "model_type": type(model).__name__,
        "features": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "model_path": str(MODEL_PATH),
        "status": "ready"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: Transaction):
    """
    Prédire si une transaction est frauduleuse
    
    Args:
        transaction: Transaction à analyser
        
    Returns:
        Prédiction avec probabilité et recommandation
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503, 
            detail="Modèle ou scaler non chargé. Veuillez réentraîner le modèle."
        )
    
    try:
        # Convertir en array
        features = np.array([[
            transaction.V1, transaction.V2, transaction.V3, transaction.V4,
            transaction.V5, transaction.V6, transaction.V7, transaction.V8,
            transaction.V9, transaction.V10, transaction.V11, transaction.V12,
            transaction.V13, transaction.V14, transaction.V15, transaction.V16,
            transaction.V17, transaction.V18, transaction.V19, transaction.V20,
            transaction.V21, transaction.V22, transaction.V23, transaction.V24,
            transaction.V25, transaction.V26, transaction.V27, transaction.V28,
            transaction.Time, transaction.Amount
        ]])
        
        # Scaling
        features_scaled = scaler.transform(features)
        
        # Prédiction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        # Interprétation
        is_fraud = bool(prediction == 1)
        
        if probability > 0.9:
            confidence = "Très élevée"
            recommendation = "🚨 Bloquer immédiatement la transaction"
        elif probability > 0.7:
            confidence = "Élevée"
            recommendation = "⚠️ Demander une vérification supplémentaire"
        elif probability > 0.5:
            confidence = "Moyenne"
            recommendation = "⚡ Surveiller la transaction de près"
        else:
            confidence = "Faible"
            recommendation = "✅ Transaction probablement légitime"
        
        return PredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=float(probability),
            confidence=confidence,
            recommendation=recommendation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: TransactionBatch):
    """
    Prédire plusieurs transactions en batch
    
    Args:
        batch: Batch de transactions
        
    Returns:
        Liste de prédictions
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle ou scaler non chargé. Veuillez réentraîner le modèle."
        )
    
    predictions = []
    frauds_count = 0
    
    for transaction in batch.transactions:
        pred = await predict(transaction)
        predictions.append(pred)
        if pred.is_fraud:
            frauds_count += 1
    
    total = len(predictions)
    fraud_rate = (frauds_count / total * 100) if total > 0 else 0
    
    return BatchPredictionResponse(
        predictions=predictions,
        total=total,
        frauds_detected=frauds_count,
        fraud_rate=fraud_rate
    )


# Point d'entrée
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

