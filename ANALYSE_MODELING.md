# 📊 Analyse du Notebook `02_modeling.ipynb`

## ✅ Points Positifs

### 1. **Structure et Organisation**
- ✅ Code bien organisé avec sections claires
- ✅ Utilisation de modules personnalisés (`preprocessing.py`, `modeling.py`)
- ✅ Commentaires personnels qui rendent le code authentique
- ✅ Pipeline de preprocessing réutilisable

### 2. **Gestion du Déséquilibre**
- ✅ Utilisation de SMOTE pour équilibrer les classes
- ✅ Split stratifié pour préserver la distribution dans train/test
- ✅ Sauvegarde du scaler pour la production

### 3. **Comparaison de Modèles**
- ✅ Comparaison systématique entre Logistic Regression et Random Forest
- ✅ Métriques complètes (Accuracy, Precision, Recall, F1, ROC-AUC)
- ✅ Visualisations d'évaluation intégrées

---

## ⚠️ Problèmes Critiques Identifiés

### 1. **Random Forest ne détecte AUCUNE fraude** 🔴

**Problème observé :**
```
Random Forest:
- TP = 0, FN = 10 (ne détecte AUCUNE fraude !)
- Precision = 0.0000
- Recall = 0.0000
- F1-Score = 0.0000
```

**Causes probables :**
- `class_weight='balanced'` peut être insuffisant avec un dataset si petit
- Le Random Forest est trop conservateur après SMOTE
- Pas d'optimisation des hyperparamètres (max_depth, min_samples_split, etc.)
- Le seuil de décision par défaut (0.5) n'est pas adapté

**Solutions :**
```python
# Option 1 : Ajuster les hyperparamètres
RandomForestClassifier(
    n_estimators=200,  # Plus d'arbres
    max_depth=15,      # Plus profond
    min_samples_split=2,  # Plus flexible
    min_samples_leaf=1,
    class_weight={0: 1, 1: 10},  # Poids personnalisé plus agressif
    random_state=42
)

# Option 2 : Optimiser le seuil de décision
from sklearn.metrics import precision_recall_curve
y_pred_proba = model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
# Trouver le seuil optimal (ex: F1-score max)
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_threshold = thresholds[np.argmax(f1_scores)]
```

### 2. **Logistic Regression : Performance très faible** 🟡

**Problème observé :**
```
Logistic Regression:
- F1-Score = 0.0072 (très faible)
- Precision = 0.0037
- Recall = 0.2000
```

**Causes :**
- Avec seulement 10 fraudes dans le test set, les métriques sont instables
- Le modèle prédit trop de faux positifs (FP=542)
- Pas d'optimisation du seuil de décision

**Solutions :**
- Utiliser une validation croisée stratifiée pour avoir plus de données de test
- Optimiser le seuil de décision avec Precision-Recall curve
- Essayer d'autres algorithmes (XGBoost, LightGBM) qui gèrent mieux le déséquilibre

### 3. **Dataset trop petit pour évaluation fiable** 🟡

**Problème :**
- 10 000 transactions totales
- Seulement 50 fraudes (0.5%)
- Test set : seulement 10 fraudes

**Impact :**
- Les métriques sont très instables
- Un seul faux négatif change drastiquement le Recall
- Difficile de généraliser les résultats

**Solutions :**
```python
# Utiliser une validation croisée stratifiée
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_idx, val_idx in skf.split(X, y):
    X_train_cv, X_val_cv = X[train_idx], X[val_idx]
    y_train_cv, y_val_cv = y[train_idx], y[val_idx]
    # ... entraînement et évaluation
```

### 4. **Pas d'optimisation des hyperparamètres** 🟡

**Problème :**
- Hyperparamètres par défaut ou basiques
- Pas de GridSearch ou RandomSearch
- Pas de tuning du seuil de décision

**Solution :**
```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

# Scorer personnalisé (F1-score)
f1_scorer = make_scorer(f1_score)

# GridSearch pour Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', {0: 1, 1: 5}, {0: 1, 1: 10}]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=StratifiedKFold(n_splits=5),
    scoring=f1_scorer,
    n_jobs=-1
)
```

### 5. **Pas d'analyse des features importantes** 🟡

**Manque :**
- Feature importance pour comprendre ce qui influence le modèle
- Visualisation des features les plus importantes
- Potentielle feature selection

**Solution :**
```python
# Après entraînement du Random Forest
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': best_detector.model.feature_importances_
}).sort_values('importance', ascending=False)

# Visualisation
plt.figure(figsize=(10, 8))
sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
plt.title('Top 15 Features les plus importantes')
plt.show()
```

### 6. **Évaluation incomplète** 🟡

**Manque :**
- Pas de courbe Precision-Recall (plus importante que ROC pour déséquilibre)
- Pas d'analyse des coûts (coût d'un faux négatif vs faux positif)
- Pas de métriques par classe détaillées

**Solution :**
```python
# Courbe Precision-Recall (plus informative que ROC pour déséquilibre)
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
ap_score = average_precision_score(y_test, y_pred_proba)

plt.plot(recall, precision, label=f'AP = {ap_score:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
```

---

## 🔧 Recommandations d'Amélioration

### Priorité 1 : Corriger le Random Forest

1. **Ajuster les hyperparamètres manuellement :**
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight={0: 1, 1: 20},  # Poids plus agressif
    random_state=42,
    n_jobs=-1
)
```

2. **Optimiser le seuil de décision :**
```python
# Trouver le seuil optimal pour maximiser F1-score
y_pred_proba = model.predict_proba(X_test)[:, 1]
f1_scores = []
thresholds = np.arange(0.1, 0.9, 0.01)

for threshold in thresholds:
    y_pred_thresh = (y_pred_proba >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred_thresh)
    f1_scores.append(f1)

optimal_threshold = thresholds[np.argmax(f1_scores)]
print(f"Seuil optimal : {optimal_threshold:.3f}")
```

### Priorité 2 : Améliorer l'évaluation

1. **Utiliser une validation croisée :**
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    model, X_train, y_train,
    cv=cv,
    scoring='f1',
    n_jobs=-1
)
print(f"F1-Score CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
```

2. **Ajouter des métriques business :**
```python
# Coût d'un faux négatif (fraude non détectée) vs faux positif
cost_fn = 100  # Coût d'une fraude non détectée
cost_fp = 1    # Coût d'une transaction bloquée à tort

total_cost = (fn * cost_fn) + (fp * cost_fp)
print(f"Coût total : {total_cost}")
```

### Priorité 3 : Essayer d'autres algorithmes

1. **XGBoost avec scale_pos_weight :**
```python
import xgboost as xgb

# Calculer le ratio de déséquilibre
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)
```

2. **LightGBM :**
```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    is_unbalance=True,  # Gère automatiquement le déséquilibre
    random_state=42
)
```

### Priorité 4 : Améliorer le preprocessing

1. **Feature Engineering :**
```python
# Créer des features dérivées
df['Amount_log'] = np.log1p(df['Amount'])
df['Time_hour'] = df['Time'] % (24 * 3600) / 3600
df['V_sum'] = df[['V1', 'V2', 'V3']].sum(axis=1)
```

2. **Feature Selection :**
```python
from sklearn.feature_selection import SelectKBest, f_classif

# Sélectionner les K meilleures features
selector = SelectKBest(f_classif, k=20)
X_selected = selector.fit_transform(X_train, y_train)
```

---

## 📝 Code d'Amélioration Suggéré

Voici un exemple de code amélioré pour la section de modélisation :

```python
# 1. Optimisation du seuil de décision
def find_optimal_threshold(y_true, y_pred_proba):
    """Trouve le seuil optimal pour maximiser F1-score"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx], f1_scores[optimal_idx]

# 2. Random Forest amélioré
rf_improved = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight={0: 1, 1: 20},  # Poids plus agressif
    random_state=42,
    n_jobs=-1
)

rf_improved.fit(X_train, y_train)
y_pred_proba_rf = rf_improved.predict_proba(X_test)[:, 1]

# Trouver le seuil optimal
optimal_threshold, optimal_f1 = find_optimal_threshold(y_test, y_pred_proba_rf)
y_pred_rf_optimized = (y_pred_proba_rf >= optimal_threshold).astype(int)

print(f"Seuil optimal : {optimal_threshold:.3f}")
print(f"F1-Score avec seuil optimal : {f1_score(y_test, y_pred_rf_optimized):.4f}")

# 3. Validation croisée
from sklearn.model_selection import cross_validate

cv_results = cross_validate(
    rf_improved, X_train, y_train,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring=['f1', 'precision', 'recall', 'roc_auc'],
    return_train_score=True
)

print(f"\n📊 Résultats CV :")
print(f"F1-Score : {cv_results['test_f1'].mean():.4f} (+/- {cv_results['test_f1'].std() * 2:.4f})")
print(f"Precision : {cv_results['test_precision'].mean():.4f}")
print(f"Recall : {cv_results['test_recall'].mean():.4f}")
```

---

## 🎯 Résumé des Actions Prioritaires

1. ✅ **Corriger Random Forest** : Ajuster hyperparamètres et seuil de décision
2. ✅ **Validation croisée** : Pour avoir des métriques plus fiables
3. ✅ **Optimisation du seuil** : Ne pas utiliser 0.5 par défaut
4. ✅ **Essayer XGBoost/LightGBM** : Meilleure gestion du déséquilibre
5. ✅ **Feature importance** : Comprendre ce qui influence le modèle
6. ✅ **Métriques business** : Coûts des erreurs

---

## 💡 Note Finale

Le code est bien structuré et professionnel, mais les performances actuelles sont insuffisantes pour un système de production. Les principales améliorations à apporter concernent :

1. **L'optimisation des hyperparamètres** (surtout pour Random Forest)
2. **L'optimisation du seuil de décision** (crucial pour le déséquilibre)
3. **L'utilisation de validation croisée** (pour des métriques plus fiables)
4. **L'essai d'autres algorithmes** (XGBoost, LightGBM)

Avec ces améliorations, vous devriez obtenir des résultats beaucoup plus satisfaisants ! 🚀

