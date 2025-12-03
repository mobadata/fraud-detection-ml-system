"""
Dashboard Streamlit pour la détection de fraude

Interface interactive pour tester le modèle de détection de fraude
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import plotly.graph_objects as go
import plotly.express as px

# Ajouter le dossier src au path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from explainability import FraudExplainer
from lime.lime_tabular import LimeTabularExplainer

# Configuration de la page
st.set_page_config(
    page_title="Détection de Fraude - Dashboard",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Chemins des modèles
MODEL_PATH = Path(__file__).parent.parent / "models" / "best_model_logistic.pkl"
SCALER_PATH = Path(__file__).parent.parent / "models" / "scaler.pkl"
DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "creditcard.csv"


@st.cache_resource
def load_model_and_scaler():
    """Charge le modèle, le scaler et l'explainer SHAP (avec cache)"""
    try:
        if MODEL_PATH.exists() and SCALER_PATH.exists():
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            
            # Charger un échantillon de données pour SHAP
            explainer = None
            if DATA_PATH.exists():
                df_sample = pd.read_csv(DATA_PATH).sample(n=min(200, len(pd.read_csv(DATA_PATH))), random_state=42)
                X_sample = df_sample.drop('Class', axis=1)
                X_sample_scaled = scaler.transform(X_sample)
                explainer = FraudExplainer(model, X_sample_scaled)
            
            return model, scaler, explainer, None
        else:
            error = f"Modèle ou scaler non trouvé. Veuillez entraîner le modèle d'abord."
            return None, None, None, error
    except Exception as e:
        return None, None, None, f"Erreur : {str(e)}"


@st.cache_data
def load_data():
    """Charge les données (avec cache)"""
    try:
        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH)
            return df, None
        else:
            return None, "Dataset non trouvé"
    except Exception as e:
        return None, f"Erreur : {str(e)}"


def predict_fraud(model, scaler, features):
    """
    Prédit si une transaction est frauduleuse
    
    Args:
        model: Modèle de ML
        scaler: Scaler pour normaliser
        features: Array de features (30 valeurs)
        
    Returns:
        prediction, probability
    """
    features_scaled = scaler.transform(features.reshape(1, -1))
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    return prediction, probability


# Titre principal
st.title("🔐 Système de Détection de Fraude")
st.markdown("---")

# Chargement du modèle et des données
model, scaler, explainer, model_error = load_model_and_scaler()
df, data_error = load_data()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Statut du système
    st.subheader("📊 Statut du système")
    if model is not None and scaler is not None:
        st.success("✅ Modèle chargé")
        st.success("✅ Scaler chargé")
    else:
        st.error(f"❌ {model_error}")
    
    if df is not None:
        st.success(f"✅ Dataset chargé ({len(df):,} transactions)")
    else:
        st.warning(f"⚠️ {data_error}")
    
    st.markdown("---")
    
    # Mode de test
    st.subheader("🎮 Mode")
    mode = st.radio(
        "Choisissez un mode",
        ["🎲 Test avec données réelles", "✏️ Saisie manuelle", "📊 Analyse dataset"]
    )

# Onglets principaux
if mode == "🎲 Test avec données réelles":
    st.header("🎲 Test avec données réelles")
    st.markdown("Sélectionnez une transaction du dataset pour tester le modèle")
    
    if df is not None and model is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Filtrer par type de transaction
            trans_type = st.selectbox(
                "Type de transaction",
                ["Toutes", "Normales uniquement", "Fraudes uniquement"]
            )
            
            if trans_type == "Normales uniquement":
                df_filtered = df[df['Class'] == 0]
            elif trans_type == "Fraudes uniquement":
                df_filtered = df[df['Class'] == 1]
            else:
                df_filtered = df
            
            # Sélection d'une transaction
            if len(df_filtered) > 0:
                idx = st.slider("Index de la transaction", 0, len(df_filtered)-1, 0)
                transaction = df_filtered.iloc[idx]
                
                # Afficher les infos
                st.subheader("📋 Détails de la transaction")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Montant", f"{transaction['Amount']:.2f} €")
                with col_b:
                    st.metric("Temps", f"{transaction['Time']:.0f} s")
                with col_c:
                    actual_label = "Fraude" if transaction['Class'] == 1 else "Normale"
                    st.metric("Vraie classe", actual_label)
                
                # Bouton de prédiction
                if st.button("🔍 Analyser cette transaction", key="predict_btn"):
                    # Extraire les features
                    feature_cols = [col for col in df.columns if col != 'Class']
                    features = transaction[feature_cols].values  # Garder 1D pour predict_fraud
                    
                    # Prédiction
                    pred, proba = predict_fraud(model, scaler, features)
                    
                    # Affichage des résultats
                    st.markdown("---")
                    st.subheader("🎯 Résultats de l'analyse")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if pred == 1:
                            st.error("🚨 **FRAUDE DÉTECTÉE**")
                        else:
                            st.success("✅ **Transaction Normale**")
                    
                    with col2:
                        st.metric("Probabilité de fraude", f"{proba*100:.2f}%")
                    
                    with col3:
                        # Vérification
                        is_correct = (pred == transaction['Class'])
                        if is_correct:
                            st.success("✅ Prédiction correcte")
                        else:
                            st.error("❌ Prédiction incorrecte")
                    
                    # Gauge de probabilité
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=proba*100,
                        title={'text': "Risque de fraude (%)"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkred" if proba > 0.5 else "green"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommandation
                    st.subheader("💡 Recommandation")
                    if proba > 0.9:
                        st.error("🚨 **Bloquer immédiatement** - Risque très élevé")
                    elif proba > 0.7:
                        st.warning("⚠️ **Vérification supplémentaire requise** - Risque élevé")
                    elif proba > 0.5:
                        st.info("⚡ **Surveillance accrue** - Risque modéré")
                    else:
                        st.success("✅ **Autoriser la transaction** - Risque faible")
                    
                    # Explicabilité avec LIME
                    st.markdown("---")
                    st.subheader("🔍 Explicabilité LIME - Pourquoi cette prédiction ?")
                    st.markdown("**LIME** (Local Interpretable Model-agnostic Explanations) explique cette prédiction spécifique")
                    
                    with st.spinner("Calcul des explications LIME..."):
                        try:
                            # Créer l'explainer LIME
                            lime_explainer = LimeTabularExplainer(
                                training_data=np.zeros((10, len(feature_cols))),  # Dummy data
                                feature_names=feature_cols,
                                class_names=['Normal', 'Fraude'],
                                mode='classification'
                            )
                            
                            # Expliquer la prédiction
                            exp = lime_explainer.explain_instance(
                                data_row=features.flatten(),
                                predict_fn=lambda x: model.predict_proba(scaler.transform(x)),
                                num_features=10
                            )
                            
                            # Extraire les features importantes
                            lime_list = exp.as_list()
                            lime_df = pd.DataFrame(lime_list, columns=['Feature', 'Impact'])
                            lime_df = lime_df.sort_values('Impact', key=abs, ascending=False)
                            
                            # Afficher le tableau
                            st.markdown("**Top 10 Features influentes selon LIME**")
                            st.dataframe(lime_df, use_container_width=True)
                            
                            # Graphique
                            fig = go.Figure(go.Bar(
                                x=lime_df['Impact'],
                                y=lime_df['Feature'],
                                orientation='h',
                                marker=dict(
                                    color=['red' if x > 0 else 'blue' for x in lime_df['Impact']],
                                )
                            ))
                            fig.update_layout(
                                title="Impact des features sur la prédiction",
                                xaxis_title="Impact (+ = vers Fraude, - = vers Normal)",
                                yaxis_title="Feature",
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.info("📊 **Comment lire** :\n"
                                   "- 🔴 **Barres rouges (positives)** : Poussent vers la FRAUDE\n"
                                   "- 🔵 **Barres bleues (négatives)** : Poussent vers NORMAL\n"
                                   "- Plus la barre est longue, plus l'influence est forte")
                            
                        except Exception as e:
                            st.warning(f"⚠️ Explications LIME non disponibles : {str(e)}")
            else:
                st.warning("Aucune transaction disponible avec ce filtre")
        
        with col2:
            # Statistiques du dataset
            st.subheader("📊 Statistiques")
            total = len(df)
            frauds = df['Class'].sum()
            normal = total - frauds
            fraud_rate = (frauds / total) * 100
            
            st.metric("Total transactions", f"{total:,}")
            st.metric("Transactions normales", f"{normal:,}")
            st.metric("Fraudes", f"{frauds:,}")
            st.metric("Taux de fraude", f"{fraud_rate:.3f}%")
    else:
        st.error("Modèle ou données non disponibles")

elif mode == "✏️ Saisie manuelle":
    st.header("✏️ Saisie manuelle d'une transaction")
    st.markdown("Entrez les valeurs des features pour tester une transaction personnalisée")
    
    if model is not None:
        st.warning("⚠️ Les features V1-V28 sont des composantes PCA anonymisées. Valeurs typiques : entre -5 et 5")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Features principales")
            time_val = st.number_input("Time (secondes)", value=0.0, step=1.0)
            amount_val = st.number_input("Amount (€)", value=100.0, min_value=0.0, step=1.0)
        
        with col2:
            st.subheader("Génération aléatoire")
            if st.button("🎲 Générer des valeurs aléatoires"):
                st.session_state['random_values'] = np.random.randn(28) * 2  # Features V1-V28
        
        st.subheader("Features V1-V28")
        st.info("💡 Conseil : Utilisez le bouton 'Générer des valeurs aléatoires' pour avoir des valeurs réalistes")
        
        # Créer un expander pour les features V
        with st.expander("🔧 Configurer les features V1-V28", expanded=False):
            v_features = []
            cols = st.columns(4)
            for i in range(28):
                col_idx = i % 4
                with cols[col_idx]:
                    default_val = st.session_state.get('random_values', np.zeros(28))[i]
                    v_val = st.number_input(f"V{i+1}", value=float(default_val), step=0.1, format="%.2f")
                    v_features.append(v_val)
        
        # Bouton de prédiction
        if st.button("🔍 Analyser cette transaction", key="predict_manual"):
            # Créer le vecteur de features
            features = np.array(v_features + [time_val, amount_val])
            
            # Prédiction
            pred, proba = predict_fraud(model, scaler, features)
            
            # Affichage
            st.markdown("---")
            st.subheader("🎯 Résultats")
            
            col1, col2 = st.columns(2)
            with col1:
                if pred == 1:
                    st.error("🚨 **FRAUDE DÉTECTÉE**")
                else:
                    st.success("✅ **Transaction Normale**")
            
            with col2:
                st.metric("Probabilité de fraude", f"{proba*100:.2f}%")
            
            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba*100,
                title={'text': "Risque de fraude (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred" if proba > 0.5 else "green"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            # Explicabilité avec LIME
            st.markdown("---")
            st.subheader("🔍 Explicabilité LIME - Pourquoi cette prédiction ?")
            
            with st.spinner("Calcul des explications LIME..."):
                try:
                    feature_names = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
                    
                    # Créer l'explainer LIME
                    lime_explainer = LimeTabularExplainer(
                        training_data=np.zeros((10, 30)),
                        feature_names=feature_names,
                        class_names=['Normal', 'Fraude'],
                        mode='classification'
                    )
                    
                    # Expliquer la prédiction
                    exp = lime_explainer.explain_instance(
                        data_row=features.flatten(),
                        predict_fn=lambda x: model.predict_proba(scaler.transform(x)),
                        num_features=10
                    )
                    
                    # Extraire les features importantes
                    lime_list = exp.as_list()
                    lime_df = pd.DataFrame(lime_list, columns=['Feature', 'Impact'])
                    lime_df = lime_df.sort_values('Impact', key=abs, ascending=False)
                    
                    # Afficher
                    st.markdown("**Top 10 Features influentes**")
                    st.dataframe(lime_df, use_container_width=True)
                    
                    # Graphique
                    fig = go.Figure(go.Bar(
                        x=lime_df['Impact'],
                        y=lime_df['Feature'],
                        orientation='h',
                        marker=dict(color=['red' if x > 0 else 'blue' for x in lime_df['Impact']])
                    ))
                    fig.update_layout(
                        title="Impact des features",
                        xaxis_title="Impact (+ = Fraude, - = Normal)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("📊 🔴 Rouge = Fraude | 🔵 Bleu = Normal")
                    
                except Exception as e:
                    st.warning(f"⚠️ Explications LIME non disponibles : {str(e)}")
    else:
        st.error("Modèle non disponible")

elif mode == "📊 Analyse dataset":
    st.header("📊 Analyse du dataset")
    
    if df is not None:
        tab1, tab2, tab3 = st.tabs(["📈 Vue d'ensemble", "💰 Analyse montants", "⏱️ Analyse temporelle"])
        
        with tab1:
            st.subheader("Vue d'ensemble du dataset")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total transactions", f"{len(df):,}")
            with col2:
                frauds = df['Class'].sum()
                st.metric("Fraudes", f"{frauds:,}")
            with col3:
                fraud_rate = (frauds / len(df)) * 100
                st.metric("Taux de fraude", f"{fraud_rate:.3f}%")
            with col4:
                avg_amount = df['Amount'].mean()
                st.metric("Montant moyen", f"{avg_amount:.2f} €")
            
            # Distribution des classes
            st.subheader("Distribution des classes")
            class_counts = df['Class'].value_counts()
            fig = px.pie(
                values=class_counts.values,
                names=['Normal', 'Fraude'],
                title="Répartition Normal vs Fraude",
                color_discrete_sequence=['green', 'red']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Analyse des montants")
            
            # Stats
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Transactions normales**")
                normal_amounts = df[df['Class'] == 0]['Amount']
                st.write(normal_amounts.describe())
            with col2:
                st.markdown("**Fraudes**")
                fraud_amounts = df[df['Class'] == 1]['Amount']
                st.write(fraud_amounts.describe())
            
            # Histogrammes
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df[df['Class'] == 0]['Amount'],
                name='Normal',
                opacity=0.7,
                marker_color='green'
            ))
            fig.add_trace(go.Histogram(
                x=df[df['Class'] == 1]['Amount'],
                name='Fraude',
                opacity=0.7,
                marker_color='red'
            ))
            fig.update_layout(
                title="Distribution des montants par classe",
                xaxis_title="Montant (€)",
                yaxis_title="Fréquence",
                barmode='overlay'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Analyse temporelle")
            
            df_temp = df.copy()
            df_temp['Time_hours'] = df_temp['Time'] / 3600
            
            # Distribution temporelle
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df_temp[df_temp['Class'] == 0]['Time_hours'],
                name='Normal',
                opacity=0.7,
                marker_color='green',
                nbinsx=50
            ))
            fig.add_trace(go.Histogram(
                x=df_temp[df_temp['Class'] == 1]['Time_hours'],
                name='Fraude',
                opacity=0.7,
                marker_color='red',
                nbinsx=50
            ))
            fig.update_layout(
                title="Distribution temporelle des transactions",
                xaxis_title="Temps (heures)",
                yaxis_title="Nombre de transactions",
                barmode='overlay'
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Dataset non disponible")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>🔐 Système de Détection de Fraude | Version 1.0.0</p>
        <p>Développé avec ❤️ par Moussa Ba</p>
    </div>
    """,
    unsafe_allow_html=True
)

