"""
Étapes 8 et 9 : intervalles entre achats, variables explicatives et séparation
train / test temporelle (notebook §8 et §9).

Principe des « photos » (snapshots) : à une date s, les variables n'utilisent que
les achats datés <= s ; la cible est ce qui se passe après s. Les photos de
test sont postérieures aux photos d'entraînement, avec une pause de HORIZON
jours : aucune information du futur n'est utilisée.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

VARIABLES = [
    "jours_depuis_achat", "intervalle_moyen", "intervalle_median", "intervalle_sd", "dernier_intervalle",
    "indice_retard", "n_achats", "frequence_mois", "taux_presence", "part_5_dernieres", "commandes_depuis_achat",
    "position_panier", "cl_commandes", "cl_ecart_moyen", "cl_ecart_sd", "cl_jours_depuis_cmd", "cl_indice_retard",
    "cl_produits_distincts", "cl_taux_reachat", "cl_taille_panier", "taille_foyer",
    "pr_taux_reachat", "pr_acheteurs", "pr_intervalle_median", "rayon",
]

DESCRIPTION = {
    "jours_depuis_achat": "jours depuis le dernier achat du produit",
    "intervalle_moyen": "intervalle moyen entre deux achats du produit",
    "intervalle_median": "intervalle médian entre deux achats du produit",
    "intervalle_sd": "écart-type des intervalles (régularité)",
    "dernier_intervalle": "dernier intervalle observé",
    "indice_retard": "jours depuis l'achat ÷ intervalle moyen (> 1 : en retard)",
    "n_achats": "nombre d'achats du produit",
    "frequence_mois": "achats du produit par mois",
    "taux_presence": "part des commandes du client contenant le produit",
    "part_5_dernieres": "part des 5 dernières commandes contenant le produit",
    "commandes_depuis_achat": "commandes passées depuis le dernier achat du produit",
    "position_panier": "rang moyen d'ajout au panier",
    "cl_commandes": "nombre de commandes du client",
    "cl_ecart_moyen": "jours moyens entre deux commandes du client",
    "cl_ecart_sd": "régularité des commandes du client",
    "cl_jours_depuis_cmd": "jours depuis la dernière commande du client",
    "cl_indice_retard": "jours depuis la dernière commande ÷ écart moyen",
    "cl_produits_distincts": "nombre de produits différents achetés",
    "cl_taux_reachat": "part de réachats du client",
    "cl_taille_panier": "produits par commande du client",
    "taille_foyer": "taille du foyer (EIA RECS 2024)",
    "pr_taux_reachat": "taux de réachat du produit (tous clients)",
    "pr_acheteurs": "nombre d'acheteurs du produit",
    "pr_intervalle_median": "intervalle médian du produit (tous clients)",
    "rayon": "rayon du produit (code)",
    "retard_vs_produit": "jours depuis l'achat ÷ intervalle médian du produit chez les autres clients",
}


def evenements(df: pd.DataFrame) -> pd.DataFrame:
    """Un événement = un produit acheté par un client à une commande, avec l'intervalle depuis l'achat précédent."""
    ev = df[["user_id", "product_id", "order_number", "jour", "add_to_cart_order", "reordered",
             "department_id", "department", "product_name"]].sort_values(["user_id", "product_id", "jour"]).reset_index(drop=True)
    ev["intervalle"] = ev.groupby(["user_id", "product_id"])["jour"].diff()
    return ev


def habitudes(ev: pd.DataFrame) -> pd.DataFrame:
    """Volet 2 : habitudes de consommation (couples achetés au moins 3 fois) et leur régularité."""
    hab = ev.groupby(["user_id", "product_id"]).agg(
        achats=("jour", "size"), intervalle_moyen=("intervalle", "mean"), intervalle_sd=("intervalle", "std"),
        rayon=("department", "first"), produit=("product_name", "first")).reset_index()
    hab = hab[hab.achats >= 3].copy()
    hab["cv"] = hab["intervalle_sd"] / hab["intervalle_moyen"].clip(lower=1)
    hab["profil"] = pd.cut(hab["cv"], [-0.01, 0.5, 1.0, np.inf], labels=["très régulier", "régulier", "irrégulier"])
    return hab


def construire_variables(s, ev, o, clients_df, avec_cible=True, achats_min=2, achats_max=None) -> pd.DataFrame:
    """achats_min=2 : modèle principal (habitudes connues). achats_min=achats_max=1 : produits achetés une seule fois (démarrage à froid)."""
    h, oh = ev[ev.jour <= s], o[o.jour <= s]
    cle = ["user_id", "product_id"]
    g = h.groupby(cle)
    f = g.agg(n_achats=("jour", "size"), premier=("jour", "min"), dernier=("jour", "max"),
              intervalle_moyen=("intervalle", "mean"), intervalle_median=("intervalle", "median"),
              intervalle_sd=("intervalle", "std"), dernier_intervalle=("intervalle", "last"),
              position_panier=("add_to_cart_order", "mean"), derniere_cmd=("order_number", "max"),
              premiere_cmd=("order_number", "min"), rayon=("department_id", "first")).reset_index()
    f = f[(f.n_achats >= achats_min) & (f.n_achats <= (achats_max or np.inf))]
    uc = oh.groupby("user_id").agg(cl_commandes=("order_id", "size"), cl_ecart_moyen=("days_since_prior_order", "mean"),
                                   cl_ecart_sd=("days_since_prior_order", "std"), cl_derniere=("jour", "max"),
                                   cl_num_cmd=("order_number", "max")).reset_index()
    ua = h.groupby("user_id").agg(cl_lignes=("product_id", "size"), cl_produits_distincts=("product_id", "nunique"),
                                  cl_taux_reachat=("reordered", "mean")).reset_index()
    uc = uc.merge(ua, on="user_id")
    uc["cl_taille_panier"] = uc["cl_lignes"] / uc["cl_commandes"]
    f = f.merge(uc, on="user_id")
    rec = h[h.order_number > h.user_id.map(uc.set_index("user_id")["cl_num_cmd"]) - 5].groupby(cle).size().rename("part_5_dernieres")
    f = f.merge(rec, left_on=cle, right_index=True, how="left")
    f["part_5_dernieres"] = f["part_5_dernieres"].fillna(0) / 5
    f["jours_depuis_achat"] = s - f["dernier"]
    f["cl_jours_depuis_cmd"] = s - f["cl_derniere"]
    f["indice_retard"] = f["jours_depuis_achat"] / f["intervalle_moyen"].clip(lower=1)
    f["cl_indice_retard"] = f["cl_jours_depuis_cmd"] / f["cl_ecart_moyen"].clip(lower=1)
    f["commandes_depuis_achat"] = f["cl_num_cmd"] - f["derniere_cmd"]
    f["taux_presence"] = f["n_achats"] / (f["cl_num_cmd"] - f["premiere_cmd"] + 1)
    f["frequence_mois"] = (f["n_achats"] - 1) / (f["dernier"] - f["premier"]).clip(lower=1) * 30
    p = h.groupby("product_id").agg(pr_taux_reachat=("reordered", "mean"), pr_acheteurs=("user_id", "nunique"),
                                    pr_intervalle_median=("intervalle", "median")).reset_index()
    f = f.merge(p, on="product_id", how="left").merge(clients_df[["taille_foyer"]], left_on="user_id", right_index=True)
    f["snapshot"] = s
    if avec_cible:
        futur = ev[ev.jour > s].groupby(cle)["jour"].min().rename("prochain")
        f = f.merge(futur, left_on=cle, right_index=True, how="left")
        d = f["prochain"] - s
        f["jours_avant_prochain_achat"] = d
        f["censure"] = d.isna() | (d > C.HORIZON)
        f["achat_sous_7j"] = (d <= C.FENETRE).fillna(False).astype(int)
    return f


def photos():
    test = [365 - C.HORIZON - C.PAS_SNAPSHOT * i for i in range(C.N_SNAP_TEST)][::-1]
    train = [test[0] - C.HORIZON - C.PAS_SNAPSHOT * i for i in range(C.N_SNAP_TRAIN)][::-1]
    return train, test


def jeu(snaps, ev, o, clients_df) -> pd.DataFrame:
    d = pd.concat([construire_variables(s, ev, o, clients_df) for s in snaps], ignore_index=True)
    d[VARIABLES] = d[VARIABLES].astype(float).fillna(-1)
    return d


# ---------------------------------------------------------------------------
# Démarrage à froid : un produit acheté une seule fois n'a pas encore d'intervalle personnel.
# On s'appuie sur l'expérience des autres clients (intervalle médian du produit, taux de réachat)
# et sur le rythme du client.
# ---------------------------------------------------------------------------
VARIABLES_FROID = [v for v in VARIABLES if v not in (
    "intervalle_moyen", "intervalle_median", "intervalle_sd", "dernier_intervalle", "indice_retard",
    "n_achats", "frequence_mois")] + ["retard_vs_produit"]


def jeu_froid(snaps, ev, o, clients_df) -> pd.DataFrame:
    d = pd.concat([construire_variables(s, ev, o, clients_df, achats_min=1, achats_max=1) for s in snaps], ignore_index=True)
    d["retard_vs_produit"] = d["jours_depuis_achat"] / d["pr_intervalle_median"].clip(lower=1)
    d[VARIABLES_FROID] = d[VARIABLES_FROID].astype(float).fillna(-1)
    return d
