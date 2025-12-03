"""
Module de modélisation pour la détection de fraude

Implémente plusieurs modèles :
- Logistic Regression (baseline)
- Random Forest
- XGBoost
- LightGBM

Avec évaluation complète et sauvegarde des modèles
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import time

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class FraudDetector:
    """
    Classe pour entraîner et évaluer des modèles de détection de fraude
    """
    
    def __init__(self, model_type='random_forest', random_state=42):
        """
        Args:
            model_type: Type de modèle ('logistic', 'random_forest', 'xgboost', 'lightgbm')
            random_state: Seed pour reproductibilité
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.metrics = {}
        self.training_time = 0
        
        # Initialiser le modèle
        self._init_model()
        
    def _init_model(self):
        """Initialise le modèle selon le type choisi"""
        if self.model_type == 'logistic':
            self.model = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=self.random_state,
                class_weight={0: 1, 1: 20},  # Poids plus agressif pour les fraudes
                n_jobs=-1
            )
        elif self.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost n'est pas installé. Installez-le avec : pip install xgboost")
            # Calculer scale_pos_weight si y_train disponible
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False,
                scale_pos_weight=1  # Sera ajusté lors de l'entraînement si nécessaire
            )
        elif self.model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM n'est pas installé. Installez-le avec : pip install lightgbm")
            self.model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                verbose=-1,
                is_unbalance=True  # Gère automatiquement le déséquilibre
            )
        else:
            raise ValueError(f"Type de modèle '{self.model_type}' non supporté")
            
        print(f"✅ Modèle initialisé : {self.model_type}")
        
    def train(self, X_train, y_train):
        """
        Entraîne le modèle
        
        Args:
            X_train: Features d'entraînement
            y_train: Target d'entraînement
        """
        print(f"\n🔧 Entraînement du modèle {self.model_type}...")
        print(f"   Shape : {X_train.shape}")
        
        # Ajuster scale_pos_weight pour XGBoost si nécessaire
        if self.model_type == 'xgboost' and hasattr(self.model, 'scale_pos_weight'):
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            self.model.set_params(scale_pos_weight=scale_pos_weight)
            print(f"   scale_pos_weight ajusté : {scale_pos_weight:.2f}")
        
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time
        
        print(f"✅ Entraînement terminé en {self.training_time:.2f}s")
        
    def predict(self, X):
        """Prédit les classes"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Prédit les probabilités"""
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X_test, y_test, show_plots=True):
        """
        Évalue le modèle sur le set de test
        
        Args:
            X_test: Features de test
            y_test: Target de test
            show_plots: Si True, affiche les graphiques
            
        Returns:
            metrics: Dictionnaire avec toutes les métriques
        """
        print(f"\n📊 ÉVALUATION du modèle {self.model_type}")
        print("="*60)
        
        # Prédictions
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        # Métriques de base
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'training_time': self.training_time
        }
        
        # Affichage
        print(f"\n🎯 Métriques :")
        print(f"   Accuracy  : {self.metrics['accuracy']:.4f}")
        print(f"   Precision : {self.metrics['precision']:.4f}")
        print(f"   Recall    : {self.metrics['recall']:.4f}")
        print(f"   F1-Score  : {self.metrics['f1_score']:.4f}")
        print(f"   ROC-AUC   : {self.metrics['roc_auc']:.4f}")
        
        # Rapport de classification
        print(f"\n📋 Classification Report :")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraude']))
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n🔢 Matrice de Confusion :")
        print(f"   TN={cm[0,0]}, FP={cm[0,1]}")
        print(f"   FN={cm[1,0]}, TP={cm[1,1]}")
        
        if show_plots:
            self._plot_evaluation(y_test, y_pred, y_pred_proba, cm)
        
        return self.metrics
    
    def _plot_evaluation(self, y_test, y_pred, y_pred_proba, cm):
        """Crée les visualisations d'évaluation"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Matrice de confusion
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0],
                    xticklabels=['Normal', 'Fraude'],
                    yticklabels=['Normal', 'Fraude'])
        axes[0,0].set_title('Matrice de Confusion')
        axes[0,0].set_ylabel('Vraie Classe')
        axes[0,0].set_xlabel('Classe Prédite')
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        axes[0,1].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {self.metrics["roc_auc"]:.3f})')
        axes[0,1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curve')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        axes[1,0].plot(recall, precision, linewidth=2)
        axes[1,0].set_xlabel('Recall')
        axes[1,0].set_ylabel('Precision')
        axes[1,0].set_title('Precision-Recall Curve')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Distribution des probabilités prédites
        axes[1,1].hist(y_pred_proba[y_test == 0], bins=50, alpha=0.7, label='Normal', color='green', edgecolor='black')
        axes[1,1].hist(y_pred_proba[y_test == 1], bins=50, alpha=0.7, label='Fraude', color='red', edgecolor='black')
        axes[1,1].set_xlabel('Probabilité prédite de fraude')
        axes[1,1].set_ylabel('Fréquence')
        axes[1,1].set_title('Distribution des Probabilités')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def save_model(self, path):
        """Sauvegarde le modèle"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"✅ Modèle sauvegardé : {path}")
        
    def load_model(self, path):
        """Charge un modèle sauvegardé"""
        self.model = joblib.load(path)
        print(f"✅ Modèle chargé : {path}")


def find_optimal_threshold(y_true, y_pred_proba):
    """
    Trouve le seuil optimal pour maximiser F1-score
    
    Args:
        y_true: Vraies valeurs
        y_pred_proba: Probabilités prédites
        
    Returns:
        optimal_threshold: Seuil optimal
        optimal_f1: F1-score au seuil optimal
    """
    from sklearn.metrics import precision_recall_curve, f1_score
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Calculer F1-score pour chaque seuil
    f1_scores = []
    for threshold in thresholds:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred_thresh, zero_division=0)
        f1_scores.append(f1)
    
    # Trouver le seuil avec le meilleur F1-score
    optimal_idx = np.argmax(f1_scores) if len(f1_scores) > 0 else 0
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    optimal_f1 = f1_scores[optimal_idx] if len(f1_scores) > 0 else 0.0
    
    return optimal_threshold, optimal_f1


def compare_models(X_train, X_test, y_train, y_test, models_to_test=None, optimize_threshold=True):
    """
    Compare plusieurs modèles
    
    Args:
        X_train, X_test, y_train, y_test: Données
        models_to_test: Liste des modèles à tester (None = tous)
        
    Returns:
        results_df: DataFrame avec les résultats comparés
    """
    if models_to_test is None:
        models_to_test = ['logistic', 'random_forest']
        if XGBOOST_AVAILABLE:
            models_to_test.append('xgboost')
        if LIGHTGBM_AVAILABLE:
            models_to_test.append('lightgbm')
    
    print("="*60)
    print("🔬 COMPARAISON DE MODÈLES")
    print("="*60)
    
    results = []
    
    for model_type in models_to_test:
        print(f"\n{'='*60}")
        print(f"   Modèle : {model_type.upper()}")
        print(f"{'='*60}")
        
        detector = FraudDetector(model_type=model_type)
        detector.train(X_train, y_train)
        
        # Obtenir les probabilités
        y_pred_proba = detector.predict_proba(X_test)
        
        # Optimiser le seuil si demandé
        if optimize_threshold:
            optimal_threshold, optimal_f1 = find_optimal_threshold(y_test, y_pred_proba)
            y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)
            
            # Recalculer les métriques avec le seuil optimal
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, 
                f1_score, roc_auc_score
            )
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred_optimized),
                'precision': precision_score(y_test, y_pred_optimized, zero_division=0),
                'recall': recall_score(y_test, y_pred_optimized),
                'f1_score': f1_score(y_test, y_pred_optimized),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'training_time': detector.training_time,
                'optimal_threshold': optimal_threshold
            }
            
            print(f"\n🎯 Seuil optimal : {optimal_threshold:.4f}")
            print(f"   F1-Score avec seuil optimal : {metrics['f1_score']:.4f}")
        else:
            metrics = detector.evaluate(X_test, y_test, show_plots=False)
        
        results.append({
            'Model': model_type,
            **metrics
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('f1_score', ascending=False)
    
    print("\n" + "="*60)
    print("📊 RÉSULTATS COMPARÉS")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Trouver le meilleur modèle
    best_model = results_df.iloc[0]['Model']
    best_f1 = results_df.iloc[0]['f1_score']
    print(f"\n🏆 Meilleur modèle : {best_model.upper()} (F1-Score = {best_f1:.4f})")
    
    return results_df


if __name__ == "__main__":
    # Test rapide
    from preprocessing import FraudPreprocessor
    
    print("🧪 Test du module de modélisation...")
    
    # Charger et préprocesser les données
    data_path = Path(__file__).parent.parent / "data" / "raw" / "creditcard.csv"
    df = pd.read_csv(data_path)
    
    preprocessor = FraudPreprocessor(random_state=42)
    X_train, X_test, y_train, y_test = preprocessor.full_pipeline(df, use_smote=True)
    
    # Tester un modèle
    print("\n" + "="*60)
    print("Test avec Random Forest")
    print("="*60)
    
    detector = FraudDetector(model_type='random_forest')
    detector.train(X_train, y_train)
    detector.evaluate(X_test, y_test, show_plots=False)
    
    print("\n✅ Test réussi !")



