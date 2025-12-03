# 🧪 Guide de Test - Fraud Detection System

## 📋 Prérequis

Assurez-vous que les dépendances sont installées :

```bash
pip install -r requirements.txt
```

---

## 🚀 1. Lancer l'API FastAPI

### Terminal 1 - API

```bash
cd /Users/moussaba/Desktop/portfolio-Moussa/fraud-detection-ml-system
python3 -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Vérification** : Vous devriez voir :
```
INFO:     Uvicorn running on http://0.0.0.0:8000
✅ Modèle chargé : /Users/.../models/best_model_random_forest.pkl
✅ Scaler chargé : /Users/.../models/scaler.pkl
```

### Accès à l'API

- **Health Check** : http://localhost:8000/
- **Documentation Swagger** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc

---

## 🎨 2. Lancer le Dashboard Streamlit

### Terminal 2 - Dashboard

```bash
cd /Users/moussaba/Desktop/portfolio-Moussa/fraud-detection-ml-system
python3 -m streamlit run streamlit_app/app.py
```

**Vérification** : Le navigateur s'ouvre automatiquement sur http://localhost:8501

---

## 🧪 3. Tester l'API

### Méthode 1 : Script Python

```bash
python3 test_api.py
```

**Résultat attendu** :
```
🧪 TEST DE L'API DE DÉTECTION DE FRAUDE
============================================================

🔍 Test 1: Health Check
   Status: 200
   Response: {'status': 'online', ...}

🔍 Test 2: Model Info
   Status: 200
   Model: RandomForestClassifier
   Features: 30

🔍 Test 3: Prediction
   Status: 200
   Fraude: ✅ NON
   Probabilité: 5.23%
   Confiance: Faible
   Recommandation: ✅ Transaction probablement légitime

============================================================
✅ TOUS LES TESTS SONT PASSÉS !
```

### Méthode 2 : cURL

```bash
# Test 1: Health Check
curl http://localhost:8000/

# Test 2: Model Info
curl http://localhost:8000/model_info

# Test 3: Prédiction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
    "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
    "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
    "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
    "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
    "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
    "Time": 406.0,
    "Amount": 149.62
  }'
```

### Méthode 3 : Swagger UI

1. Ouvrir http://localhost:8000/docs
2. Cliquer sur `/predict` → **Try it out**
3. Modifier les valeurs dans le JSON
4. Cliquer sur **Execute**
5. Voir le résultat dans **Response body**

### Méthode 4 : Python Requests

```python
import requests

url = "http://localhost:8000/predict"
transaction = {
    "V1": -1.36, "V2": -0.07, # ... toutes les features
    "Time": 406.0,
    "Amount": 149.62
}

response = requests.post(url, json=transaction)
result = response.json()

print(f"Fraude: {result['is_fraud']}")
print(f"Probabilité: {result['fraud_probability']:.2%}")
```

---

## 🎨 4. Tester le Dashboard

### Scénarios de test

#### Scénario 1 : Test avec données réelles

1. Ouvrir http://localhost:8501
2. Mode : **🎲 Test avec données réelles**
3. Type : **Fraudes uniquement**
4. Sélectionner une transaction avec le slider
5. Cliquer sur **🔍 Analyser cette transaction**
6. **Vérifier** : Prédiction vs Vraie classe

#### Scénario 2 : Saisie manuelle

1. Mode : **✏️ Saisie manuelle**
2. Cliquer sur **🎲 Générer des valeurs aléatoires**
3. Modifier **Amount** (ex: 5000€ pour tester un gros montant)
4. Cliquer sur **🔍 Analyser cette transaction**
5. **Observer** : Jauge de probabilité et recommandation

#### Scénario 3 : Analyse du dataset

1. Mode : **📊 Analyse dataset**
2. Onglet **Vue d'ensemble** : Voir les statistiques
3. Onglet **Analyse montants** : Comparer fraudes vs normales
4. Onglet **Analyse temporelle** : Distribution dans le temps

---

## 🐳 5. Tester avec Docker

### Option 1 : Docker Compose

```bash
docker-compose up -d

# Vérifier les logs
docker-compose logs -f

# Accéder aux services
# API: http://localhost:8000
# Dashboard: http://localhost:8501

# Arrêter
docker-compose down
```

### Option 2 : Docker manuel

```bash
# API
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api

# Dashboard
docker build -f Dockerfile.streamlit -t fraud-dashboard .
docker run -p 8501:8501 fraud-dashboard
```

---

## 📊 6. Scénarios de test complets

### Test 1 : Transaction normale (petit montant)
```json
{
  "Amount": 15.50,
  "Time": 100.0,
  "V1": 0.5, "V2": -0.3, ... (valeurs proches de 0)
}
```
**Attendu** : Fraude = NON, Probabilité < 10%

### Test 2 : Transaction suspecte (gros montant + features anormales)
```json
{
  "Amount": 5000.0,
  "Time": 50000.0,
  "V1": -5.2, "V2": 4.8, "V4": 6.5, ... (valeurs extrêmes)
}
```
**Attendu** : Fraude = OUI, Probabilité > 70%

---

## ✅ Checklist de validation

- [ ] API se lance sans erreur
- [ ] Health check retourne `status: online`
- [ ] Model info retourne les bonnes infos
- [ ] Prédiction unique fonctionne
- [ ] Dashboard se lance sans erreur
- [ ] Mode "données réelles" fonctionne
- [ ] Mode "saisie manuelle" fonctionne
- [ ] Graphiques s'affichent correctement
- [ ] Docker build réussit
- [ ] Docker compose up fonctionne

---

## 🐛 Troubleshooting

### Erreur : "ModuleNotFoundError: No module named 'fastapi'"

```bash
pip install -r requirements.txt
```

### Erreur : "Modèle non trouvé"

```bash
# Réentraîner le modèle
jupyter notebook notebooks/02_modeling.ipynb
# Exécuter toutes les cellules
```

### Erreur : "Port 8000 already in use"

```bash
# Trouver et tuer le processus
lsof -ti:8000 | xargs kill -9

# Ou utiliser un autre port
uvicorn api.main:app --port 8001
```

### Dashboard ne se lance pas

```bash
# Vérifier streamlit
pip install --upgrade streamlit

# Vérifier les dépendances
pip install plotly
```

---

## 📞 Support

Si vous rencontrez des problèmes :

1. Vérifier les logs des services
2. Consulter le fichier USAGE.md
3. Créer une issue sur GitHub

---

**Bon test ! 🚀**




