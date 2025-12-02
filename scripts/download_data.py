"""
Script pour télécharger les données de détection de fraude
"""
import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import RAW_DATA_DIR
import urllib.request
import zipfile
import os

def download_data():
    """
    Télécharge le dataset Credit Card Fraud Detection
    
    Note: Pour utiliser l'API Kaggle, vous devez:
    1. Créer un compte sur Kaggle
    2. Aller dans Account > Create New API Token
    3. Placer le fichier kaggle.json dans ~/.kaggle/
    
    Alternative: Télécharger manuellement depuis:
    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    """
    
    print("🔍 Vérification du dataset...")
    
    data_file = RAW_DATA_DIR / "creditcard.csv"
    
    if data_file.exists():
        print(f"✅ Dataset déjà présent: {data_file}")
        print(f"📊 Taille du fichier: {data_file.stat().st_size / 1024 / 1024:.2f} MB")
        return
    
    print("\n📥 Téléchargement du dataset...")
    print("=" * 60)
    
    # Méthode 1: Via Kaggle API (recommandé)
    try:
        import kaggle
        print("🔑 Authentification Kaggle détectée")
        print("📦 Téléchargement en cours...")
        
        kaggle.api.dataset_download_files(
            'mlg-ulb/creditcardfraud',
            path=str(RAW_DATA_DIR),
            unzip=True
        )
        
        print("✅ Dataset téléchargé avec succès!")
        return
        
    except Exception as e:
        print(f"⚠️ Kaggle API non configurée: {e}")
        print("\n" + "=" * 60)
        print("📋 INSTRUCTIONS POUR CONFIGURER KAGGLE API:")
        print("=" * 60)
        print("1. Créez un compte sur https://www.kaggle.com")
        print("2. Allez dans 'Account' > 'Create New API Token'")
        print("3. Téléchargez le fichier kaggle.json")
        print("4. Placez-le dans ~/.kaggle/ (créez le dossier si nécessaire)")
        print("5. Sur Linux/Mac: chmod 600 ~/.kaggle/kaggle.json")
        print("6. Installez kaggle: pip install kaggle")
        print("\n" + "=" * 60)
        print("📥 ALTERNATIVE: Téléchargement manuel")
        print("=" * 60)
        print("1. Visitez: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("2. Cliquez sur 'Download'")
        print(f"3. Extrayez creditcard.csv dans: {RAW_DATA_DIR}")
        print("=" * 60)
        
        # Méthode 2: Dataset alternatif (plus petit, pour test)
        print("\n💡 Génération d'un dataset de démonstration...")
        generate_demo_data()

def generate_demo_data():
    """
    Génère un dataset de démonstration pour tester le pipeline
    """
    import pandas as pd
    import numpy as np
    
    print("🔄 Génération de données synthétiques...")
    
    np.random.seed(42)
    
    # Nombre d'échantillons
    n_samples = 10000
    n_frauds = 50  # 0.5% de fraudes
    
    # Générer des features (simulant les composantes PCA)
    data = {
        **{f'V{i}': np.random.randn(n_samples) for i in range(1, 29)},
        'Time': np.random.randint(0, 172800, n_samples),
        'Amount': np.random.lognormal(3, 2, n_samples),
        'Class': np.zeros(n_samples)
    }
    
    # Marquer certaines transactions comme frauduleuses
    fraud_indices = np.random.choice(n_samples, n_frauds, replace=False)
    data['Class'][fraud_indices] = 1
    
    # Modifier légèrement les features pour les fraudes (patterns différents)
    for idx in fraud_indices:
        for i in range(1, 15):
            data[f'V{i}'][idx] += np.random.randn() * 2
        data['Amount'][idx] *= np.random.uniform(1.5, 3.0)
    
    # Créer DataFrame
    df = pd.DataFrame(data)
    
    # Sauvegarder
    output_path = RAW_DATA_DIR / "creditcard.csv"
    df.to_csv(output_path, index=False)
    
    print(f"✅ Dataset de démonstration créé: {output_path}")
    print(f"📊 Nombre de transactions: {len(df):,}")
    print(f"🚨 Nombre de fraudes: {n_frauds} ({n_frauds/len(df)*100:.2f}%)")
    print(f"💰 Montant moyen: {df['Amount'].mean():.2f}€")
    print("\n⚠️ NOTE: Ceci est un dataset de DÉMONSTRATION")
    print("Pour de vrais résultats, utilisez le dataset Kaggle original")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🔐 TÉLÉCHARGEMENT DU DATASET - DÉTECTION DE FRAUDE")
    print("=" * 60 + "\n")
    
    download_data()
    
    print("\n" + "=" * 60)
    print("✅ TERMINÉ!")
    print("=" * 60)
    print("\n🚀 Prochaines étapes:")
    print("1. Explorez les données: jupyter notebook notebooks/01_data_exploration.ipynb")
    print("2. Entraînez le modèle: python src/train.py")
    print("3. Lancez l'API: uvicorn src.api.main:app --reload")
    print("=" * 60 + "\n")

