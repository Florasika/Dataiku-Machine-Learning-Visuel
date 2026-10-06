# 🤖 Jour 3 / 10 — Dataiku : Machine Learning Visuel

> **Série : 10 Days of Dataiku** · Jour 3/10  
> Concepts : Lab · AutoML · Feature Engineering · Classification · Déploiement de modèle

---

## 📁 Fichiers du projet

```
day-03-machine-learning/
│
├── dataset_ml_dataiku_j3.xlsx   ← 351 lignes · 17 features · variable cible Churn
├── recipe_ml_j3.py              ← Script Python reproduisant l'AutoML
└── README.md
```

---

## 🧠 Objectif du modèle

```
Prédire quels clients risquent de churner (se désabonner)
à partir de leur historique d'achats.

Variable cible : Churn (0 = reste, 1 = part)
Taux de churn  : ~23% dans le dataset
Type de modèle : Classification binaire
```

---

## 🚀 ÉTAPE 1 — Importer le dataset

```
Dans Dataiku :
→ Datasets → + Dataset → Upload → dataset_ml_dataiku_j3.xlsx
→ Nommer "dataset_ml"
→ Vérifier dans Explore que la colonne "Churn" contient 0 et 1
```

---

## 🔑 ÉTAPE 2 — Préparer les données (Prepare Recipe)

```
Flow → dataset_ml → Recipes → Visual → Prepare

Étapes à ajouter :
1. Colonne Date → Parse date (yyyy-MM-dd)
2. Vérifier les nulls : Analyze → voir la distribution de chaque colonne
3. Churn → Rename en "target_churn"

→ Output : "dataset_ml_propre"
→ Run
```

---

## 🔑 ÉTAPE 3 — Créer un modèle dans le Lab

```
Flow → cliquer sur "dataset_ml_propre"
→ Lab (icône fiole en haut à droite du dataset)
→ "+ New Analysis" → "Machine Learning"

Configuration :
→ Target variable   : Churn
→ Type              : Classification (auto-détecté car 0/1)
→ Cliquer "Create"
```

---

## 🔑 ÉTAPE 4 — Configurer les features

```
Onglet "Features" :

Inclure :
✓ Montant
✓ Marge
✓ Taux_Marge
✓ Quantité
✓ Ancienneté_Ans
✓ Nb_Contacts
✓ Mois_Num
✓ Catégorie       → type : Categorical
✓ Segment_Client  → type : Categorical
✓ Région          → type : Categorical
✓ Vendeur         → type : Categorical

Exclure (ne pas cocher) :
✗ ID           (identifiant unique, pas prédictif)
✗ Date         (redondant avec Mois_Num)
✗ Mois
✗ Trimestre
✗ Prix_Unitaire (corrélé avec Montant)

→ Dataiku encode automatiquement les variables catégorielles
```

---

## 🔑 ÉTAPE 5 — Configurer l'AutoML

```
Onglet "Algorithms" :

Cocher les algorithmes à comparer :
✓ Random Forest
✓ Gradient Boosted Trees (XGBoost)
✓ Logistic Regression

Onglet "Metrics" :
→ Optimization metric : AUC (ROC)
→ Test set : 20% (hold-out)
→ Cross-validation : 5 folds

→ Cliquer "Train" en haut à droite
→ Dataiku lance les 3 modèles en parallèle
```

---

## 🔑 ÉTAPE 6 — Analyser les résultats

```
Après l'entraînement, Dataiku affiche :

Tableau comparatif :
┌────────────────────────┬──────────┬───────┐
│ Modèle                 │ Accuracy │  AUC  │
├────────────────────────┼──────────┼───────┤
│ Gradient Boosted Trees │   84.3%  │ 0.891 │
│ Random Forest          │   82.1%  │ 0.876 │
│ Logistic Regression    │   78.5%  │ 0.831 │
└────────────────────────┴──────────┴───────┘

→ Cliquer sur le meilleur modèle pour l'analyse détaillée
```

---

## 🔑 ÉTAPE 7 — Interpréter le meilleur modèle

```
Dans l'analyse du modèle sélectionné :

Onglet "Performance" :
→ Courbe ROC           : visualiser l'AUC
→ Matrice de confusion  : vrais/faux positifs
→ Seuil de décision     : ajuster selon le métier

Onglet "Variable Importance" :
→ Montant          ████████████  32%
→ Ancienneté_Ans   ████████      18%
→ Taux_Marge       ██████        14%
→ Nb_Contacts      █████         12%
→ Segment_Client   ████          10%

Onglet "Partial Dependencies" :
→ Voir l'effet de chaque feature sur la prédiction
→ Montant élevé → probabilité de churn basse ✓
→ Ancienneté basse → probabilité de churn haute ✓
```

---

## 🔑 ÉTAPE 8 — Déployer le modèle dans le Flow

```
Sur le meilleur modèle :
→ "Deploy" → "Create Flow Model"
→ Nommer : "modele_churn_v1"

Le modèle apparaît maintenant dans le Flow.

Créer une recette de scoring :
→ Flow → modele_churn_v1 → "Score"
→ Input : dataset_ml_propre
→ Output : "predictions_churn" (nouveau dataset)
→ Colonnes ajoutées : proba_churn, prediction, classe

→ Run → explorer le dataset predictions_churn
```

---

## 🔑 ÉTAPE 9 — Utiliser le script Python (local)

```bash
pip install pandas scikit-learn openpyxl

python recipe_ml_j3.py

# Résultat :
# Dataset : 351 lignes | Churn rate : 23.4%
# Features : 12
#   Random Forest          → Acc: 82.1% | AUC: 0.876
#   Gradient Boosting      → Acc: 84.3% | AUC: 0.891
#   Logistic Regression    → Acc: 78.5% | AUC: 0.831
# Meilleur modèle : Gradient Boosting
# Clients à risque élevé : 10 affichés
```

---

## 🔑 ÉTAPE 10 — Analyser les prédictions

```
Dans Dataiku → dataset "predictions_churn" → Explore

Graphique risque :
→ Charts → Bar chart
→ X : Risque (Faible/Moyen/Élevé)
→ Y : COUNT(*)

→ Voir combien de clients sont en risque élevé

Filtre risque élevé :
→ Filtrer sur prediction = 1
→ Group by Vendeur → voir qui a le plus de clients à risque
→ Alerter les vendeurs concernés
```

---

## 💡 AutoML Dataiku vs Scikit-learn

| | AutoML Dataiku | Scikit-learn |
|---|---|---|
| **Code** | Aucun | Python |
| **Vitesse** | Très rapide | Dépend |
| **Comparaison modèles** | Automatique | Manuel |
| **Interprétabilité** | Interface visuelle | Graphiques Python |
| **Déploiement** | 1 clic | Code API |
| **Reproductibilité** | Versioning intégré | Git |

---



---

⭐ **Si ce projet t'aide, mets une étoile !**
