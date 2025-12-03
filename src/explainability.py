"""
Module d'explicabilité avec SHAP

Permet d'expliquer les prédictions du modèle de détection de fraude
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class FraudExplainer:
    """
    Classe pour expliquer les prédictions avec SHAP
    """
    
    def __init__(self, model, X_train_sample=None):
        """
        Args:
            model: Modèle ML entraîné
            X_train_sample: Échantillon de données d'entraînement pour SHAP
        """
        self.model = model
        self.explainer = None
        self.shap_values = None
        
        # Initialiser l'explainer si on a un échantillon
        if X_train_sample is not None:
            self._init_explainer(X_train_sample)
    
    def _init_explainer(self, X_train_sample):
        """Initialise l'explainer SHAP"""
        print("🔧 Initialisation de SHAP...")
        
        # Utiliser TreeExplainer pour RF/XGBoost ou LinearExplainer pour LR
        model_type = type(self.model).__name__
        
        if 'Tree' in model_type or 'Forest' in model_type or 'XGB' in model_type or 'LightGBM' in model_type:
            # TreeExplainer pour les modèles à base d'arbres
            self.explainer = shap.TreeExplainer(self.model)
            print(f"✅ TreeExplainer initialisé pour {model_type}")
        else:
            # KernelExplainer pour les autres (Logistic Regression, etc.)
            # Prendre un petit échantillon pour accélérer
            if len(X_train_sample) > 100:
                background = shap.sample(X_train_sample, 100)
            else:
                background = X_train_sample
            
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, 
                background
            )
            print(f"✅ KernelExplainer initialisé pour {model_type}")
    
    def explain_prediction(self, X, feature_names=None):
        """
        Explique une prédiction unique
        
        Args:
            X: Features de la transaction (array 1D ou 2D)
            feature_names: Noms des features
            
        Returns:
            shap_values, expected_value
        """
        if self.explainer is None:
            raise ValueError("Explainer non initialisé. Passez X_train_sample au constructeur.")
        
        # S'assurer que X est 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Calculer les SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # Pour les classifieurs binaires, prendre les valeurs de la classe positive
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
        
        return shap_values, self.explainer.expected_value
    
    def get_top_features(self, shap_values, feature_names, top_n=10):
        """
        Récupère les top N features les plus importantes
        
        Args:
            shap_values: SHAP values (1D array)
            feature_names: Noms des features
            top_n: Nombre de features à retourner
            
        Returns:
            DataFrame avec features et leur importance
        """
        # Valeurs absolues pour l'importance
        importance = np.abs(shap_values)
        
        # Créer un DataFrame
        df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP_Value': shap_values.flatten(),
            'Importance': importance.flatten()
        })
        
        # Trier par importance
        df = df.sort_values('Importance', ascending=False).head(top_n)
        
        return df
    
    def plot_waterfall(self, shap_values, feature_values, feature_names, max_display=10):
        """
        Crée un waterfall plot pour expliquer une prédiction
        
        Args:
            shap_values: SHAP values (1D)
            feature_values: Valeurs des features (1D)
            feature_names: Noms des features
            max_display: Nombre max de features à afficher
        """
        # S'assurer que tout est 1D
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
        if len(feature_values.shape) > 1:
            feature_values = feature_values[0]
        
        # Créer l'explanation object
        explanation = shap.Explanation(
            values=shap_values,
            base_values=self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
            data=feature_values,
            feature_names=feature_names
        )
        
        # Plot
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_force(self, shap_values, feature_values, feature_names):
        """
        Crée un force plot pour visualiser l'explication
        
        Args:
            shap_values: SHAP values (1D)
            feature_values: Valeurs des features (1D)
            feature_names: Noms des features
        """
        # S'assurer que tout est 1D
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
        if len(feature_values.shape) > 1:
            feature_values = feature_values[0]
        
        # Force plot
        return shap.force_plot(
            base_value=self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
            shap_values=shap_values,
            features=feature_values,
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
    
    def plot_bar(self, shap_values, feature_names, max_display=10):
        """
        Crée un bar plot des features les plus importantes
        
        Args:
            shap_values: SHAP values (peut être 1D ou 2D)
            feature_names: Noms des features
            max_display: Nombre max de features à afficher
        """
        # S'assurer que c'est 1D
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
        
        # Trier par importance absolue
        importance = np.abs(shap_values)
        indices = np.argsort(importance)[::-1][:max_display]
        
        # Préparer les données
        features = [feature_names[i] for i in indices]
        values = shap_values[indices]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['red' if v > 0 else 'blue' for v in values]
        ax.barh(features, values, color=colors, alpha=0.7)
        ax.set_xlabel('Impact sur la prédiction (SHAP value)', fontsize=12)
        ax.set_title('Top Features - Contribution à la Prédiction', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        # Inverser l'ordre pour avoir le plus important en haut
        ax.invert_yaxis()
        
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    # Test rapide
    from preprocessing import FraudPreprocessor
    from modeling import FraudDetector
    import pandas as pd
    from pathlib import Path
    
    print("🧪 Test du module d'explicabilité...")
    
    # Charger les données
    data_path = Path(__file__).parent.parent / "data" / "raw" / "creditcard.csv"
    df = pd.read_csv(data_path)
    
    # Preprocessing
    preprocessor = FraudPreprocessor(random_state=42)
    X_train, X_test, y_train, y_test = preprocessor.full_pipeline(df, use_smote=True)
    
    # Charger le modèle
    model_path = Path(__file__).parent.parent / "models" / "best_model_logistic.pkl"
    import joblib
    model = joblib.load(model_path)
    
    # Créer l'explainer
    print("\n🔧 Création de l'explainer...")
    explainer = FraudExplainer(model, X_train[:100])  # Petit échantillon pour le test
    
    # Expliquer une prédiction
    print("\n🔍 Explication d'une prédiction...")
    X_sample = X_test[0:1]
    shap_values, expected_value = explainer.explain_prediction(X_sample, preprocessor.feature_cols)
    
    print(f"✅ SHAP values calculées : shape = {shap_values.shape}")
    
    # Top features
    top_features = explainer.get_top_features(shap_values, preprocessor.feature_cols, top_n=5)
    print("\n🔝 Top 5 features :")
    print(top_features)
    
    print("\n✅ Test réussi !")

