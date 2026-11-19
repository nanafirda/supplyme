"""
Nouveaux clients sans antécédents : le démarrage à froid (notebook §12 bis).

Trois situations, trois réponses :
  0 commande      → aucune prédiction individuelle possible : panier de bienvenue (produits les plus
                    achetés et rachetés), objectif = obtenir la 1re commande.
  1 achat         → « modèle à froid » : on prédit à partir de l'expérience des autres clients
                    (intervalle médian du produit, taux de réachat) et du rythme du client.
  2 achats et +   → modèle principal personnel ; sa précision augmente avec l'historique.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score

import config as C
from prediction.features import VARIABLES_FROID, jeu_froid


def couverture_deuxieme_commande(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Quelle part du 2e panier peut-on deviner ? Avec les best-sellers (0 commande) ou avec le 1er panier (1 commande)."""
    p1 = df[df.order_number == 1].groupby("user_id")["product_id"].apply(set)
    p2 = df[df.order_number == 2].groupby("user_id")["product_id"].apply(set)
    u = p1.index.intersection(p2.index)
    top = set(df.groupby("product_id")["user_id"].nunique().nlargest(top_n).index)
    lignes = [
        ("0 commande : les 10 best-sellers", np.mean([len(top & p2[i]) / len(p2[i]) for i in u])),
        ("1 commande : le 1er panier", np.mean([len(p1[i] & p2[i]) / len(p2[i]) for i in u])),
    ]
    return pd.DataFrame(lignes, columns=["situation", "part du 2e panier retrouvée"])


def modele_froid():
    return lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=100, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=C.SEED)


def regression_froid():
    return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, objective="l1", verbose=-1,
                             random_state=C.SEED)


def evaluer_produits_achetes_une_fois(snap_train, snap_test, ev, o, clients_df) -> pd.DataFrame:
    TR, TE = jeu_froid(snap_train, ev, o, clients_df), jeu_froid(snap_test, ev, o, clients_df)
    y, yr = TE["achat_sous_7j"], TE.loc[~TE.censure, "jours_avant_prochain_achat"]
    # règle simple : le client suit le rythme moyen des autres acheteurs du produit
    regle = -np.abs(TE["pr_intervalle_median"] - TE["jours_depuis_achat"] - 3.5)
    proba = modele_froid().fit(TR[VARIABLES_FROID], TR["achat_sous_7j"]).predict_proba(TE[VARIABLES_FROID])[:, 1]
    delai_regle = np.clip(TE.loc[~TE.censure, "pr_intervalle_median"] - TE.loc[~TE.censure, "jours_depuis_achat"], 1, C.HORIZON)
    reg = regression_froid().fit(TR.loc[~TR.censure, VARIABLES_FROID], TR.loc[~TR.censure, "jours_avant_prochain_achat"])
    delai_modele = np.clip(reg.predict(TE.loc[~TE.censure, VARIABLES_FROID]), 1, C.HORIZON)
    return pd.DataFrame({
        "AUC (achat sous 7 j)": [roc_auc_score(y, regle), roc_auc_score(y, proba)],
        "Average Precision": [average_precision_score(y, regle), average_precision_score(y, proba)],
        "MAE délai (j)": [mean_absolute_error(yr, delai_regle), mean_absolute_error(yr, delai_modele)],
        "observations test": [len(TE), len(TE)],
        "taux de base": [y.mean(), y.mean()],
    }, index=["Règle : rythme des autres clients", "Modèle à froid (LightGBM)"])


def precision_selon_historique(TE: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    """AUC du modèle principal selon le nombre de commandes déjà passées par le client."""
    t = TE.assign(p=proba, historique=pd.cut(TE["cl_commandes"], [0, 3, 5, 10, 20, 1000],
                                             labels=["1–3", "4–5", "6–10", "11–20", "21 et +"]))
    return (t.groupby("historique", observed=True)
            .apply(lambda g: pd.Series({"observations": len(g), "AUC": roc_auc_score(g["achat_sous_7j"], g["p"])}))
            .reset_index())
