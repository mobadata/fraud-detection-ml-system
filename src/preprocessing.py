"""
Preprocessing et Feature Engineering pour la détection de fraude

Ce module gère :
- Nettoyage des données
- Scaling des features
- Gestion du déséquilibre de classes (SMOTE)
- Split train/test stratifié
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path

class FraudPreprocessor:
    """
    Classe pour gérer le preprocessing des données de fraude
    
    Features:
    - Scaling standardisé
    - Gestion du déséquilibre avec SMOTE
    - Sauvegarde/Chargement du scaler
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.target_col = 'Class'
        
    def prepare_features(self, df):
        """
        Sépare features et target
        
        Args:
            df: DataFrame avec toutes les colonnes
            
        Returns:
            X, y: Features et target
        """
        # Toutes les colonnes sauf Class
        self.feature_cols = [col for col in df.columns if col != self.target_col]
        
        X = df[self.feature_cols]
        y = df[self.target_col]
        
        print(f"✅ Features préparées : {X.shape}")
        print(f"✅ Target : {y.shape}")
        
        return X, y
    
    def scale_features(self, X_train, X_test=None, fit=True):
        """
        Scale les features avec StandardScaler
        
        Args:
            X_train: Features d'entraînement
            X_test: Features de test (optionnel)
            fit: Si True, fit le scaler sur X_train
            
        Returns:
            X_train_scaled, X_test_scaled (ou None)
        """
        if fit:
            X_train_scaled = self.scaler.fit_transform(X_train)
            print("✅ Scaler fitted sur X_train")
        else:
            X_train_scaled = self.scaler.transform(X_train)
            
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled, None
    
    def handle_imbalance(self, X_train, y_train, method='smote'):
        """
        Gère le déséquilibre des classes
        
        Args:
            X_train: Features d'entraînement
            y_train: Target d'entraînement
            method: Méthode à utiliser ('smote', 'adasyn', ou None)
            
        Returns:
            X_resampled, y_resampled
        """
        if method is None:
            return X_train, y_train
        
        print(f"\n📊 Avant resampling :")
        print(f"   Classe 0 (Normal) : {(y_train == 0).sum()}")
        print(f"   Classe 1 (Fraude) : {(y_train == 1).sum()}")
        
        if method == 'smote':
            sampler = SMOTE(random_state=self.random_state)
        else:
            raise ValueError(f"Méthode '{method}' non supportée")
        
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        
        print(f"\n📊 Après SMOTE :")
        print(f"   Classe 0 (Normal) : {(y_resampled == 0).sum()}")
        print(f"   Classe 1 (Fraude) : {(y_resampled == 1).sum()}")
        print(f"✅ Resampling terminé : {X_resampled.shape}")
        
        return X_resampled, y_resampled
    
    def split_data(self, X, y, test_size=0.2, stratify=True):
        """
        Split train/test stratifié
        
        Args:
            X: Features
            y: Target
            test_size: Proportion de test
            stratify: Si True, split stratifié
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=stratify_param
        )
        
        print(f"\n✅ Split effectué :")
        print(f"   Train : {X_train.shape}")
        print(f"   Test  : {X_test.shape}")
        print(f"   Fraudes train : {(y_train == 1).sum()} ({(y_train == 1).sum() / len(y_train) * 100:.2f}%)")
        print(f"   Fraudes test  : {(y_test == 1).sum()} ({(y_test == 1).sum() / len(y_test) * 100:.2f}%)")
        
        return X_train, X_test, y_train, y_test
    
    def full_pipeline(self, df, test_size=0.2, use_smote=True):
        """
        Pipeline complet de preprocessing
        
        Args:
            df: DataFrame brut
            test_size: Proportion de test
            use_smote: Si True, applique SMOTE
            
        Returns:
            X_train, X_test, y_train, y_test (tous preprocessed)
        """
        print("="*60)
        print("🔧 PIPELINE DE PREPROCESSING")
        print("="*60)
        
        # 1. Préparer features et target
        X, y = self.prepare_features(df)
        
        # 2. Split train/test
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size=test_size)
        
        # 3. Scaling
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test, fit=True)
        
        # 4. SMOTE (uniquement sur train)
        if use_smote:
            X_train_resampled, y_train_resampled = self.handle_imbalance(
                X_train_scaled, y_train, method='smote'
            )
        else:
            X_train_resampled, y_train_resampled = X_train_scaled, y_train
            
        print("\n" + "="*60)
        print("✅ PREPROCESSING TERMINÉ")
        print("="*60)
        
        return X_train_resampled, X_test_scaled, y_train_resampled, y_test
    
    def save_scaler(self, path):
        """Sauvegarde le scaler"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path)
        print(f"✅ Scaler sauvegardé : {path}")
        
    def load_scaler(self, path):
        """Charge un scaler sauvegardé"""
        self.scaler = joblib.load(path)
        print(f"✅ Scaler chargé : {path}")


def create_feature_summary(df):
    """
    Crée un résumé des features pour documentation
    
    Args:
        df: DataFrame
        
    Returns:
        summary_df: DataFrame avec statistiques
    """
    summary = pd.DataFrame({
        'Feature': df.columns,
        'Type': df.dtypes,
        'Missing': df.isnull().sum(),
        'Missing %': (df.isnull().sum() / len(df) * 100).round(2),
        'Unique': df.nunique(),
        'Mean': df.mean(numeric_only=True),
        'Std': df.std(numeric_only=True),
        'Min': df.min(numeric_only=True),
        'Max': df.max(numeric_only=True)
    })
    
    return summary


if __name__ == "__main__":
    # Test rapide du preprocessing
    from pathlib import Path
    
    print("🧪 Test du module preprocessing...")
    
    # Charger les données
    data_path = Path(__file__).parent.parent / "data" / "raw" / "creditcard.csv"
    df = pd.read_csv(data_path)
    
    print(f"\n📊 Dataset chargé : {df.shape}")
    
    # Test du preprocessor
    preprocessor = FraudPreprocessor(random_state=42)
    X_train, X_test, y_train, y_test = preprocessor.full_pipeline(df, use_smote=True)
    
    print(f"\n✅ Test réussi !")
    print(f"   X_train final : {X_train.shape}")
    print(f"   X_test final  : {X_test.shape}")

