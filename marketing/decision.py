"""
Volets 4, 5 et 7 : moment d'envoi, décision A / B / C et rentabilité (notebook §14, §15, §17).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from marketing import economics as eco


def candidats(P: pd.DataFrame, fenetre=C.FENETRE) -> pd.DataFrame:
    """Besoins à traiter. P : une ligne par couple client × produit (proba_7j, delai_predit, marge, prix…)."""
    habitude = (P.n_achats >= 3) & (P.intervalle_sd <= P.intervalle_moyen)          # achat répété et assez régulier
    depasse = habitude & P.indice_retard.between(C.RETARD_MIN, C.RETARD_MAX) & (P.jours_depuis_achat <= C.HORIZON)
    imminent = (P.delai_predit <= fenetre) & ~depasse
    D = P[imminent | depasse].copy()
    D["segment"] = np.where(depasse[imminent | depasse], "besoin dépassé", "besoin imminent")
    D["valeur"] = eco.valeur_achat(D.marge.values, D.intervalle_moyen.values, D.segment.values)
    return D


def decider(P: pd.DataFrame, u=C.U_RELANCE, e=C.ELASTICITE, c=C.COUT_MESSAGE) -> pd.DataFrame:
    return optimiser(candidats(P), u, e, c)


def optimiser(D: pd.DataFrame, u=C.U_RELANCE, e=C.ELASTICITE, c=C.COUT_MESSAGE) -> pd.DataFrame:
    """Choix A / B / C pour chaque besoin, puis regroupement en un message par client."""
    D = D.copy()
    p0, V, R = D.proba_7j.values, D.valeur.values, D.prix.values
    D["gain_A"] = eco.gain_A(p0, V, u)
    gB = np.column_stack([eco.gain_B(p0, V, R, d, u, e) for d in C.REMISES])
    D["remise_optimale"] = np.array(C.REMISES)[gB.argmax(1)]
    D["gain_B"] = gB.max(1)
    D["option"] = np.where(D.gain_B > D.gain_A, "B — promo", "A — relance")
    D["gain"] = D[["gain_A", "gain_B"]].max(1)
    # 12 produits maximum par message : les plus rentables de chaque client
    D["pool"] = D["gain"].groupby(D.user_id).rank(ascending=False, method="first") <= C.MAX_PRODUITS_MESSAGE
    D.loc[(D.gain <= 0) | ~D.pool, "option"] = "C — rien"
    utile = D.option != "C — rien"
    total = D[utile].groupby("user_id")["gain"].sum()
    envoyer = total[total > c].index
    D["decision"] = np.where(D.user_id.isin(envoyer) & utile, D.option, "C — rien")
    actif = D.decision != "C — rien"
    n_prod = D[actif].groupby("user_id")["product_id"].transform("size")
    D["profit"] = 0.0
    D.loc[actif, "profit"] = D.loc[actif, "gain"] - c / n_prod
    D["remise"] = np.where(D.decision == "B — promo", D.remise_optimale, 0.0)
    D["proba_attendue"] = np.select([D.decision == "B — promo", D.decision == "A — relance"],
                                    [eco.proba_B(p0, D.remise.values, u, e), eco.proba_A(p0, u)], p0)
    # Volet 4 : 1 jour avant le besoin (dès demain si le besoin est dépassé), 1 h avant l'heure habituelle
    demain = C.END_DATE + pd.Timedelta(days=1)
    D["date_envoi"] = (D["date_prevue"] - pd.Timedelta(days=1)).clip(lower=demain)
    D.loc[D.segment == "besoin dépassé", "date_envoi"] = demain
    D["heure_envoi"] = (D["heure_habituelle"] - 1).clip(lower=7)
    return D


def strategies(D: pd.DataFrame, u=C.U_RELANCE, e=C.ELASTICITE, c=C.COUT_MESSAGE) -> pd.DataFrame:
    """Comparaison sur exactement les mêmes produits (les 12 besoins principaux de chaque client)."""
    D = D[D.pool]
    p0, V, R = D.proba_7j.values, D.valeur.values, D.prix.values
    n, s, d = D.user_id.nunique(), (D.decision != "C — rien").values, C.REMISE_BENCHMARK
    lignes = [
        ("Aucune relance", 0, 0.0, 0.0),
        ("Relance à tous, sans réduction", n, eco.gain_A(p0, V, u).sum() - n * c, 0.0),
        (f"Promo -{d:.0%} à tous", n, eco.gain_B(p0, V, R, d, u, e).sum() - n * c,
         (eco.proba_B(p0, d, u, e) * d * R).sum()),
        ("SupplyMe (ciblé)", D.loc[s, "user_id"].nunique(), D.profit.sum(),
         (eco.proba_B(p0[s], D.remise.values[s], u, e) * D.remise.values[s] * R[s]).sum()),
    ]
    t = pd.DataFrame(lignes, columns=["strategy", "customers_contacted", "incremental_profit", "discount_cost"])
    t["message_cost"] = t["customers_contacted"] * c
    return t


def sensibilite(P: pd.DataFrame, us=(0.0, 0.025, 0.05, 0.075, 0.10), es=(1.5, 2.0, 2.62, 3.0, 3.5)) -> pd.DataFrame:
    D = candidats(P)
    return pd.DataFrame([[optimiser(D, u, e).profit.sum() for e in es] for u in us],
                        index=[f"{u:.1%}" for u in us], columns=[str(e) for e in es])
