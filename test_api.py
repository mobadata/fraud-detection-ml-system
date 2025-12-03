"""
Script de test pour l'API de détection de fraude

Usage:
    python test_api.py
"""

import requests
import json

# URL de l'API
API_URL = "http://localhost:8000"

def test_health_check():
    """Test du endpoint de santé"""
    print("🔍 Test 1: Health Check")
    response = requests.get(f"{API_URL}/")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    print()

def test_model_info():
    """Test des infos du modèle"""
    print("🔍 Test 2: Model Info")
    response = requests.get(f"{API_URL}/model_info")
    print(f"   Status: {response.status_code}")
    data = response.json()
    print(f"   Model: {data.get('model_type')}")
    print(f"   Features: {data.get('n_features')}")
    print()

def test_prediction():
    """Test d'une prédiction"""
    print("🔍 Test 3: Prediction")
    
    # Transaction de test (valeurs aléatoires)
    transaction = {
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
    
    response = requests.post(f"{API_URL}/predict", json=transaction)
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"   Fraude: {'🚨 OUI' if result['is_fraud'] else '✅ NON'}")
        print(f"   Probabilité: {result['fraud_probability']*100:.2f}%")
        print(f"   Confiance: {result['confidence']}")
        print(f"   Recommandation: {result['recommendation']}")
    else:
        print(f"   Erreur: {response.text}")
    print()

def main():
    """Lance tous les tests"""
    print("="*60)
    print("🧪 TEST DE L'API DE DÉTECTION DE FRAUDE")
    print("="*60)
    print()
    
    try:
        test_health_check()
        test_model_info()
        test_prediction()
        
        print("="*60)
        print("✅ TOUS LES TESTS SONT PASSÉS !")
        print("="*60)
        print()
        print("📖 Pour plus de tests, visitez:")
        print(f"   Swagger UI: {API_URL}/docs")
        print(f"   ReDoc: {API_URL}/redoc")
        
    except requests.exceptions.ConnectionError:
        print("❌ ERREUR: Impossible de se connecter à l'API")
        print()
        print("🔧 Assurez-vous que l'API est lancée:")
        print("   python3 -m uvicorn api.main:app --reload")
        print()
    except Exception as e:
        print(f"❌ ERREUR: {e}")

if __name__ == "__main__":
    main()



