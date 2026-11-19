"""
Volet 5 : le modèle économique (notebook §15). Raisonnement INCRÉMENTAL.

    pA = p0 + u·(1 − p0)                    relance simple
    pB = min(pA·(1 + ε·d), 0,95)            relance + réduction d (ε = 2,62, Bijmolt et al. 2005)
    Profit_A = (pA − p0)·V − c
    Profit_B = pB·(V − d·R) − p0·V − c
    Profit_C = 0

V = valeur d'un achat gagné : la marge pour un besoin imminent ; la marge × les achats
récupérés sur 90 jours pour un besoin dépassé (raisonnement de la valeur client).
La réduction est payée à tous les acheteurs, y compris ceux qui auraient acheté
sans elle (effet d'aubaine).
"""
import numpy as np

import config as C


def proba_A(p0, u=C.U_RELANCE):
    return np.minimum(p0 + u * (1 - p0), C.P_MAX)


def proba_B(p0, d, u=C.U_RELANCE, e=C.ELASTICITE):
    return np.minimum(proba_A(p0, u) * (1 + e * d), C.P_MAX)


def gain_A(p0, V, u=C.U_RELANCE):
    return (proba_A(p0, u) - p0) * V


def gain_B(p0, V, R, d, u=C.U_RELANCE, e=C.ELASTICITE):
    return proba_B(p0, d, u, e) * (V - d * R) - p0 * V


def valeur_achat(marge, intervalle_moyen, segment):
    """Besoin dépassé : on récupère l'habitude pour HORIZON_FIDELITE jours (plafonné à 6 achats)."""
    achats = np.minimum(C.HORIZON_FIDELITE / np.clip(intervalle_moyen, 1, None), C.ACHATS_RECUPERES_MAX)
    return np.where(segment == "besoin dépassé", marge * achats, marge)
