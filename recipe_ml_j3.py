# ============================================================
#  JOUR 3 / 10 — Dataiku : Machine Learning
#  Script Python reproduisant l'AutoML Dataiku en local
#  Dans Dataiku : ce code va dans une recette Python
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, accuracy_score)
import warnings
warnings.filterwarnings('ignore')

# ── Chargement (dans Dataiku) ─────────────────────────────────
# import dataiku
# df = dataiku.Dataset("dataset_ml_enrichi").get_dataframe()

# ── En local ──────────────────────────────────────────────────
df = pd.read_excel("dataset_ml_dataiku_j3.xlsx", sheet_name="Dataset_ML")
print(f"Dataset : {len(df)} lignes | Churn rate : {df['Churn'].mean()*100:.1f}%")

# ════════════════════════════════════════════════════════════
#  ÉTAPE 1 : Préparation des features (Feature Engineering)
# ════════════════════════════════════════════════════════════
df_ml = df.copy()

# Encoder les variables catégorielles
encodeurs = {}
for col in ['Vendeur', 'Région', 'Catégorie', 'Segment_Client']:
    le = LabelEncoder()
    df_ml[f'{col}_enc'] = le.fit_transform(df_ml[col].astype(str))
    encodeurs[col] = le

# Features sélectionnées pour le modèle
FEATURES = [
    'Montant', 'Marge', 'Taux_Marge', 'Quantité',
    'Prix_Unitaire', 'Ancienneté_Ans', 'Nb_Contacts',
    'Mois_Num',
    'Vendeur_enc', 'Région_enc', 'Catégorie_enc', 'Segment_Client_enc',
]
TARGET = 'Churn'

X = df_ml[FEATURES]
y = df_ml[TARGET]

print(f"\nFeatures : {len(FEATURES)}")
print(f"Distribution cible : {y.value_counts().to_dict()}")

# ════════════════════════════════════════════════════════════
#  ÉTAPE 2 : Split train/test
# ════════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain : {len(X_train)} | Test : {len(X_test)}")

# ════════════════════════════════════════════════════════════
#  ÉTAPE 3 : Entraîner plusieurs modèles (comme l'AutoML)
# ════════════════════════════════════════════════════════════
modeles = {
    'Random Forest'       : RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting'   : GradientBoostingClassifier(random_state=42),
    'Logistic Regression' : LogisticRegression(random_state=42, max_iter=1000),
}

resultats = []
for nom, modele in modeles.items():
    modele.fit(X_train, y_train)
    y_pred  = modele.predict(X_test)
    y_proba = modele.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_proba)
    cv_auc  = cross_val_score(modele, X, y, cv=5, scoring='roc_auc').mean()

    resultats.append({
        'Modèle'  : nom,
        'Accuracy': round(acc*100, 1),
        'AUC'     : round(auc, 3),
        'CV AUC'  : round(cv_auc, 3),
    })
    print(f"  {nom:25} → Acc: {acc*100:.1f}% | AUC: {auc:.3f} | CV AUC: {cv_auc:.3f}")

# ════════════════════════════════════════════════════════════
#  ÉTAPE 4 : Meilleur modèle — analyse détaillée
# ════════════════════════════════════════════════════════════
df_resultats = pd.DataFrame(resultats).sort_values('AUC', ascending=False)
print(f"\n=== Classement des modèles ===")
print(df_resultats.to_string(index=False))

meilleur_nom = df_resultats.iloc[0]['Modèle']
meilleur     = modeles[meilleur_nom]

print(f"\n=== Meilleur modèle : {meilleur_nom} ===")
y_pred_best = meilleur.predict(X_test)

print("\nRapport de classification :")
print(classification_report(y_test, y_pred_best,
                             target_names=['Non Churn', 'Churn']))

# ════════════════════════════════════════════════════════════
#  ÉTAPE 5 : Importance des features
# ════════════════════════════════════════════════════════════
if hasattr(meilleur, 'feature_importances_'):
    importances = pd.DataFrame({
        'Feature'   : FEATURES,
        'Importance': meilleur.feature_importances_,
    }).sort_values('Importance', ascending=False)

    print("\n=== Importance des features ===")
    print(importances.head(8).to_string(index=False))

# ════════════════════════════════════════════════════════════
#  ÉTAPE 6 : Prédictions sur nouveaux clients
# ════════════════════════════════════════════════════════════
df_ml['Proba_Churn']  = meilleur.predict_proba(X)[:, 1].round(3)
df_ml['Pred_Churn']   = meilleur.predict(X)
df_ml['Risque']       = pd.cut(df_ml['Proba_Churn'],
                                bins=[0, 0.3, 0.6, 1],
                                labels=['Faible', 'Moyen', 'Élevé'])

print("\n=== Distribution du risque de churn ===")
print(df_ml['Risque'].value_counts())

# Clients à risque élevé
a_risque = df_ml[df_ml['Risque'] == 'Élevé'][
    ['ID','Vendeur','Segment_Client','Montant','Proba_Churn']
].head(10)
print("\n=== Top 10 clients à risque élevé ===")
print(a_risque.to_string(index=False))

# ── Écriture dans Dataiku ─────────────────────────────────────
# output = dataiku.Dataset("predictions_churn")
# output.write_with_schema(df_ml[['ID','Proba_Churn','Pred_Churn','Risque']])
print("\n✓ ML terminé — modèle prêt pour déploiement")
