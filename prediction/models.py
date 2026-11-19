"""
Étapes 10 à 13 : comparaison de modèles, choix du meilleur, prédiction finale
(notebook §10 à §13).

Régression   : délai avant le prochain achat (baseline, linéaire, Random Forest, XGBoost, LightGBM).
Classification : probabilité de rachat sous 7 jours (baseline, logistique, Random Forest, XGBoost, LightGBM).
"""
from __future__ import annotations

import logging
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C
from prediction.features import VARIABLES

log = logging.getLogger(__name__)


def modeles_regression():
    return {
        "Régression linéaire": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=14, min_samples_leaf=20, max_features=0.5,
                                               n_jobs=-1, random_state=C.SEED),
        "XGBoost": xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8,
                                    objective="reg:absoluteerror", tree_method="hist", n_jobs=-1, random_state=C.SEED),
        "LightGBM": lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                                      colsample_bytree=0.8, objective="l1", verbose=-1, random_state=C.SEED),
    }


def modeles_classification():
    return {
        "Régression logistique": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=14, min_samples_leaf=20, max_features=0.5,
                                                n_jobs=-1, random_state=C.SEED),
        "XGBoost": xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                                     tree_method="hist", eval_metric="auc", n_jobs=-1, random_state=C.SEED),
        "LightGBM": lgb.LGBMClassifier(n_estimators=800, learning_rate=0.03, num_leaves=63, min_child_samples=50,
                                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=C.SEED),
    }


def classifieur_optimise():
    return lgb.LGBMClassifier(verbose=-1, random_state=C.SEED, **C.LGBM_CLASSIF)


def scores_reg(y, p):
    return {"MAE (j)": mean_absolute_error(y, p), "RMSE (j)": np.sqrt(mean_squared_error(y, p)),
            "R²": r2_score(y, p), "% à ±7 j": np.mean(np.abs(y - p) <= 7) * 100}


def comparer_regression(TR, TE):
    tr, te = TR[~TR.censure], TE[~TE.censure]
    Xtr, ytr, Xte, yte = tr[VARIABLES], tr["jours_avant_prochain_achat"], te[VARIABLES], te["jours_avant_prochain_achat"]
    ech = Xtr.sample(min(C.ECHANTILLON_MODELES_LENTS, len(Xtr)), random_state=C.SEED).index
    preds = {"Baseline": np.clip(te["intervalle_median"] - te["jours_depuis_achat"], 1, C.HORIZON)}
    res = {"Baseline": scores_reg(yte, preds["Baseline"])}
    modeles = modeles_regression()
    for nom, m in modeles.items():
        t0 = time.time()
        lent = nom == "Random Forest"
        m.fit(Xtr.loc[ech] if lent else Xtr, ytr.loc[ech] if lent else ytr)
        preds[nom] = np.clip(m.predict(Xte), 1, C.HORIZON)
        res[nom] = scores_reg(yte, preds[nom])
        log.info("  régression · %-20s MAE %.2f j (%.0f s)", nom, res[nom]["MAE (j)"], time.time() - t0)
    tab = pd.DataFrame(res).T.sort_values("MAE (j)")
    tab["gain MAE vs baseline (%)"] = (1 - tab["MAE (j)"] / tab.loc["Baseline", "MAE (j)"]) * 100
    meilleur = tab.index[0]
    reel_vs_predit = pd.DataFrame({"reel": yte.values, "predit": preds[meilleur]}).sample(5000, random_state=0)
    return tab, modeles[meilleur], meilleur, reel_vs_predit


def comparer_classification(TR, TE):
    Xtr, ytr, Xte, yte = TR[VARIABLES], TR["achat_sous_7j"], TE[VARIABLES], TE["achat_sous_7j"]
    ech = Xtr.sample(min(C.ECHANTILLON_MODELES_LENTS, len(Xtr)), random_state=C.SEED).index
    probas = {"Baseline": -np.abs(TE["intervalle_median"] - TE["jours_depuis_achat"] - 3.5)}
    for nom, m in modeles_classification().items():
        t0 = time.time()
        lent = nom == "Random Forest"
        m.fit(Xtr.loc[ech] if lent else Xtr, ytr.loc[ech] if lent else ytr)
        probas[nom] = m.predict_proba(Xte)[:, 1]
        log.info("  classification · %-22s AUC %.3f (%.0f s)", nom, roc_auc_score(yte, probas[nom]), time.time() - t0)
    opt = classifieur_optimise().fit(Xtr, ytr)
    probas["LightGBM optimisé"] = opt.predict_proba(Xte)[:, 1]
    res, rocs = {}, []
    for nom, p in probas.items():
        res[nom] = {"AUC": roc_auc_score(yte, p), "Average Precision": average_precision_score(yte, p),
                    "Brier": brier_score_loss(yte, p) if nom != "Baseline" else np.nan}
        fpr, tpr, _ = roc_curve(yte, p)
        idx = np.linspace(0, len(fpr) - 1, 200).astype(int)
        rocs.append(pd.DataFrame({"modele": nom, "fpr": fpr[idx], "tpr": tpr[idx]}))
    tab = pd.DataFrame(res).T.sort_values("AUC", ascending=False)
    importance = pd.Series(opt.booster_.feature_importance("gain"), index=VARIABLES)
    importance = (importance / importance.sum() * 100).sort_values(ascending=False)
    log.info("  taux de base (achat sous 7 j, test) : %.1f %%", yte.mean() * 100)
    return tab, pd.concat(rocs, ignore_index=True), importance, float(yte.mean()), probas


def choisir_classifieur(TR: pd.DataFrame, snap_train: list) -> tuple[str, pd.DataFrame]:
    """Sélection du modèle SANS regarder le test : on entraîne sur les premières photos du train et on compare
    l'AUC sur les 3 dernières photos du train (validation), séparées par une pause de HORIZON jours."""
    val_snaps = snap_train[-3:]
    fit = TR[TR.snapshot <= val_snaps[0] - C.HORIZON]
    val = TR[TR.snapshot.isin(val_snaps)]
    ech = fit.sample(min(C.ECHANTILLON_MODELES_LENTS, len(fit)), random_state=C.SEED).index
    candidats = {**modeles_classification(), "LightGBM optimisé": classifieur_optimise()}
    res = {}
    for nom, m in candidats.items():
        lent = nom == "Random Forest"
        m.fit(fit.loc[ech, VARIABLES] if lent else fit[VARIABLES], fit.loc[ech, "achat_sous_7j"] if lent else fit["achat_sous_7j"])
        res[nom] = roc_auc_score(val["achat_sous_7j"], m.predict_proba(val[VARIABLES])[:, 1])
        log.info("  validation · %-22s AUC %.4f", nom, res[nom])
    tab = pd.Series(res, name="AUC validation").sort_values(ascending=False).to_frame()
    return tab.index[0], tab


def classifieur(nom: str):
    return {**modeles_classification(), "LightGBM optimisé": classifieur_optimise()}[nom]
