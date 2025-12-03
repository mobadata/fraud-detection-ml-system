# 🚀 Guide d'Utilisation - Fraud Detection System

## 📋 Table des matières
- [Installation](#installation)
- [Utilisation de l'API FastAPI](#api-fastapi)
- [Utilisation du Dashboard Streamlit](#dashboard-streamlit)
- [Entraînement du modèle](#entraînement)

---

## 🔧 Installation

### 1. Cloner le repo
```bash
git clone https://github.com/mobadata/fraud-detection-ml-system.git
cd fraud-detection-ml-system
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Télécharger les données
```bash
python scripts/download_data.py
```

---

## 🤖 API FastAPI

### Lancer l'API

```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

L'API sera accessible sur : `http://localhost:8000`

### Documentation interactive

- **Swagger UI** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc

### Endpoints disponibles

#### 1. Health Check
```bash
curl http://localhost:8000/
```

#### 2. Informations sur le modèle
```bash
curl http://localhost:8000/model_info
```

#### 3. Prédiction unique
```bash
curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "V1": -1.3598071336738,
    "V2": -0.0727811733098497,
    "V3": 2.53634673796914,
    ... (toutes les features V1-V28)
    "Time": 406.0,
    "Amount": 149.62
  }'
```

**Réponse** :
```json
{
  "is_fraud": false,
  "fraud_probability": 0.05,
  "confidence": "Faible",
  "recommendation": "✅ Transaction probablement légitime"
}
```

#### 4. Prédictions en batch
```bash
curl -X POST "http://localhost:8000/predict_batch" \\
  -H "Content-Type: application/json" \\
  -d '{
    "transactions": [
      { "V1": ..., "V2": ..., ... },
      { "V1": ..., "V2": ..., ... }
    ]
  }'
```

### Exemple en Python

```python
import requests

# URL de l'API
url = "http://localhost:8000/predict"

# Transaction à tester
transaction = {
    "V1": -1.3598071336738,
    "V2": -0.0727811733098497,
    "V3": 2.53634673796914,
    # ... autres features ...
    "Time": 406.0,
    "Amount": 149.62
}

# Envoyer la requête
response = requests.post(url, json=transaction)
result = response.json()

print(f"Fraude : {result['is_fraud']}")
print(f"Probabilité : {result['fraud_probability']:.2%}")
print(f"Recommandation : {result['recommendation']}")
```

---

## 🎨 Dashboard Streamlit

### Lancer le Dashboard

```bash
streamlit run streamlit_app/app.py
```

Le dashboard sera accessible sur : `http://localhost:8501`

### Fonctionnalités

#### 1. 🎲 Test avec données réelles
- Sélectionner une transaction du dataset
- Voir les détails (montant, temps, vraie classe)
- Analyser avec le modèle
- Comparer prédiction vs réalité
- Visualisation du risque avec une jauge

#### 2. ✏️ Saisie manuelle
- Entrer manuellement les valeurs des features
- Générer des valeurs aléatoires réalistes
- Tester des transactions personnalisées
- Voir la probabilité de fraude

#### 3. 📊 Analyse dataset
- Vue d'ensemble du dataset
- Statistiques par classe (normal vs fraude)
- Distribution des montants
- Analyse temporelle
- Graphiques interactifs

### Captures d'écran

#### Prédiction en temps réel
![Dashboard](docs/images/dashboard_prediction.png)

#### Analyse du dataset
![Analytics](docs/images/dashboard_analytics.png)

---

## 🎓 Entraînement du modèle

### 1. Exploration des données
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 2. Entraînement
```bash
jupyter notebook notebooks/02_modeling.ipynb
```

Ou en ligne de commande :
```python
from src.preprocessing import FraudPreprocessor
from src.modeling import FraudDetector
import pandas as pd

# Charger les données
df = pd.read_csv('data/raw/creditcard.csv')

# Preprocessing
preprocessor = FraudPreprocessor(random_state=42)
X_train, X_test, y_train, y_test = preprocessor.full_pipeline(df, use_smote=True)

# Entraîner le modèle
detector = FraudDetector(model_type='random_forest')
detector.train(X_train, y_train)
detector.evaluate(X_test, y_test)

# Sauvegarder
detector.save_model('models/best_model_random_forest.pkl')
preprocessor.save_scaler('models/scaler.pkl')
```

---

## 🐳 Docker (À venir)

### Construire l'image
```bash
docker build -t fraud-detection-api .
```

### Lancer le container
```bash
docker run -p 8000:8000 fraud-detection-api
```

---

## 📊 Métriques du Modèle

| Modèle | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| Random Forest | 0.9995 | 0.95 | 0.82 | 0.88 | 0.99 |
| XGBoost | 0.9996 | 0.96 | 0.85 | 0.90 | 0.99 |
| LightGBM | 0.9997 | 0.97 | 0.87 | 0.92 | 0.99 |

---

## 🔒 Sécurité & Production

### Recommandations pour la production :

1. **Authentification** : Ajouter un système d'authentification (JWT, OAuth2)
2. **Rate limiting** : Limiter le nombre de requêtes par utilisateur
3. **Logging** : Logger toutes les prédictions pour audit
4. **Monitoring** : Surveiller les performances du modèle
5. **Retraining** : Réentraîner régulièrement avec de nouvelles données
6. **A/B Testing** : Tester plusieurs modèles en parallèle
7. **Explicabilité** : Ajouter SHAP/LIME pour expliquer les prédictions

---

## 📞 Support

Pour toute question ou problème :
- **GitHub Issues** : [Créer une issue](https://github.com/mobadata/fraud-detection-ml-system/issues)
- **Email** : moussa.ba@example.com

---

## 📄 Licence

MIT License - Voir [LICENSE](LICENSE) pour plus de détails.

---

**Développé avec ❤️ par Moussa Ba**

