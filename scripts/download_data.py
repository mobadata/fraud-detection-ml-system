# Script pour télécharger les données
# Bon, en gros ce script vérifie si les données existent déjà,
# sinon il essaye de les télécharger depuis Kaggle
# Si Kaggle marche pas, je génère des données de démo pour tester

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import RAW_DATA_DIR
import pandas as pd
import numpy as np

def download_data():
    data_file = RAW_DATA_DIR / "creditcard.csv"
    
    # Check si déjà téléchargé
    if data_file.exists():
        print(f"OK, le dataset est déjà là: {data_file}")
        size_mb = data_file.stat().st_size / 1024 / 1024
        print(f"Taille: {size_mb:.1f} MB")
        return
    
    print("Les données ne sont pas là, faut les télécharger...")
    
    # Essayer avec l'API Kaggle
    try:
        import kaggle
        print("Cool, Kaggle API configuré, téléchargement...")
        kaggle.api.dataset_download_files(
            'mlg-ulb/creditcardfraud',
            path=str(RAW_DATA_DIR),
            unzip=True
        )
        print("Téléchargement terminé !")
        return
    except:
        # Tant pis, Kaggle marche pas
        print("Kaggle API pas configuré...")
        print("\nSi vous voulez le vrai dataset:")
        print("1. Allez sur kaggle.com et créez un compte")
        print("2. Account > Create New API Token")
        print("3. Mettez kaggle.json dans ~/.kaggle/")
        print("4. pip install kaggle")
        print("\nOu téléchargez manuellement: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print(f"Et mettez le fichier dans: {RAW_DATA_DIR}")
        
        # Générer des données de test
        print("\nBon, je vais créer des données de démo pour tester...")
        generate_demo_data()

def generate_demo_data():
    # Générer des données synthétiques pour tester le pipeline
    # C'est pas les vraies données mais ça permet de développer
    
    print("Génération de données synthétiques...")
    np.random.seed(42)
    
    n_samples = 10000
    n_frauds = 50  # environ 0.5% comme dans le vrai dataset
    
    # Créer les features (V1-V28 sont des composantes PCA dans le vrai dataset)
    data = {}
    for i in range(1, 29):
        data[f'V{i}'] = np.random.randn(n_samples)
    
    data['Time'] = np.random.randint(0, 172800, n_samples)
    data['Amount'] = np.random.lognormal(3, 2, n_samples)
    data['Class'] = np.zeros(n_samples)
    
    # Marquer quelques transactions comme fraudes
    fraud_idx = np.random.choice(n_samples, n_frauds, replace=False)
    data['Class'][fraud_idx] = 1
    
    # Modifier un peu les features pour les fraudes (pour avoir des patterns)
    for idx in fraud_idx:
        for i in range(1, 15):
            data[f'V{i}'][idx] += np.random.randn() * 2
        data['Amount'][idx] *= np.random.uniform(1.5, 3.0)
    
    df = pd.DataFrame(data)
    
    # Sauvegarder
    output_path = RAW_DATA_DIR / "creditcard.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\nDataset de démo créé: {output_path}")
    print(f"Nombre de transactions: {len(df):,}")
    print(f"Fraudes: {n_frauds} ({n_frauds/len(df)*100:.2f}%)")
    print(f"Montant moyen: {df['Amount'].mean():.2f}€")
    print("\n⚠️  Attention: données de DÉMONSTRATION uniquement")
    print("Pour de vrais résultats, utilisez le dataset Kaggle")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Téléchargement des données - Détection de Fraude")
    print("="*50 + "\n")
    
    download_data()
    
    print("\n✅ Terminé !")
    print("\nProchaines étapes:")
    print("  - Explorer les données: jupyter notebook notebooks/01_data_exploration.ipynb")
    print("  - Entraîner un modèle: python src/train.py (quand ce sera fait)")
    print()

