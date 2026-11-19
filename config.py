"""
Configuration centrale de SupplyMe.

Tous les paramètres et HYPOTHÈSES sont ici, avec leur source, pour être cités
tels quels dans le rapport. Ils sont identiques à ceux du notebook
notebooks/SupplyMe_Analyse_Complete.ipynb : le pipeline reproduit ses résultats.
"""
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
BASE_GROUPE = ROOT / "data" / "base_groupe"   # archive SupplyMe_Final_Excel_and_Original_Data décompressée
RAW_DIR = BASE_GROUPE / "original_instacart"  # les 6 fichiers Instacart d'origine
DB_PATH = ROOT / "supplyme.db"
SCHEMA_PATH = ROOT / "database" / "schema.sql"
ARTIFACTS = ROOT / "prediction" / "artifacts"
REPORTS = ROOT / "reports"

INSTACART_FILES = ["orders", "order_products__prior", "order_products__train",
                   "products", "aisles", "departments"]

# ---------------------------------------------------------------------------
# Échantillon et calendrier
# ---------------------------------------------------------------------------
SEED = 42
# Échantillon : les 3 000 vrais clients de la base du groupe (tirage aléatoire, graine 20261118)
# et ses 1 000 nouveaux comptes fictifs (questionnaire d'accueil)
END_DATE = pd.Timestamp("2026-10-31")   # fin d'historique (convention : Instacart n'a pas de dates)
DECALAGE_MAX = 13             # dernière commande de chaque client placée 0 à 13 jours avant END_DATE

# ---------------------------------------------------------------------------
# Données complémentaires (sources dans le README et le notebook §6)
# ---------------------------------------------------------------------------
# Prix moyens U.S. city average — Bureau of Labor Statistics, Average Price Data
ANCRES_BLS = [
    ("Œufs, gros, la douzaine", 2.71, "sept. 2025", "dairy eggs"),
    ("Lait entier, le gallon", 4.05, "sept. 2025", "dairy eggs"),
    ("Pain blanc, la livre", 1.87, "sept. 2025", "bakery"),
    ("Bananes, la livre", 0.65, "août 2026", "produce"),
    ("Salade iceberg, la livre", 1.47, "août 2026", "produce"),
    ("Bœuf haché, la livre", 6.92, "août 2026", "meat seafood"),
    ("Blanc de poulet, la livre", 4.17, "août 2026", "meat seafood"),
    ("Café, la livre", 6.32, "2024", "beverages"),
]
DISPERSION_PRIX = 0.35        # écart-type (log) des prix au sein d'un rayon — hypothèse
MARGE_BRUTE = 0.279           # NGA / FMS Solutions, Independent Grocers Financial Survey 2025
# Répartition des foyers américains par taille (millions) — EIA, RECS 2024, table HC9.4
FOYERS_RECS = {1: 43.74, 2: 43.87, 3: 18.84, 4: 14.60, 5: 11.50}

# ---------------------------------------------------------------------------
# Prédiction
# ---------------------------------------------------------------------------
HORIZON = 60                  # jours : au-delà, le délai est censuré
FENETRE = 7                   # cible de classification : rachat sous 7 jours
PAS_SNAPSHOT = 14
N_SNAP_TRAIN, N_SNAP_TEST = 12, 3
ECHANTILLON_MODELES_LENTS = 200_000   # Random Forest entraîné sur un sous-échantillon
# Meilleurs réglages LightGBM trouvés par validation temporelle (notebook §12)
LGBM_CLASSIF = dict(num_leaves=63, learning_rate=0.02, n_estimators=1200, min_child_samples=200,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8)

# ---------------------------------------------------------------------------
# Marketing — sources et hypothèses
# ---------------------------------------------------------------------------
U_RELANCE = 0.05              # part des hésitants convaincus par un rappel — HYPOTHÈSE (sensibilité 0–10 %)
ELASTICITE = 2.62             # Bijmolt, Van Heerde & Pieters (2005), JMR : élasticité-prix moyenne −2,62
COUT_MESSAGE = 0.05           # $ par message (routage + CRM) — HYPOTHÈSE
REMISES = [0.05, 0.10, 0.15, 0.20]
P_MAX = 0.95
HORIZON_FIDELITE = 90         # jours d'habitude récupérés si un client « décroché » est reconquis — HYPOTHÈSE
ACHATS_RECUPERES_MAX = 6
RETARD_MIN, RETARD_MAX = 1.5, 3.0   # « besoin dépassé » : 1,5 à 3 fois l'intervalle habituel sans achat
MAX_PRODUITS_MESSAGE = 12
REMISE_BENCHMARK = 0.10

# ---------------------------------------------------------------------------
# Nouveaux clients sans historique
# ---------------------------------------------------------------------------
ALPHA_GROUPE = 30               # force du lissage bayésien choisie par validation dans la base du groupe
BRUIT_REPONSES = 0.6            # bruit des réponses simulées au questionnaire — HYPOTHÈSE
PART_REPONSES_FAUSSES = 0.15    # part des clients dont une réponse ne reflète pas leurs achats — HYPOTHÈSE
POINTS_PAR_REPONSE = 10         # points de fidélité par rayon indiqué au questionnaire
BONUS_QUESTIONNAIRE_COMPLET = 50
VALEUR_POINT = 0.01             # 100 points = 1 $ de réduction sur la 1re commande
