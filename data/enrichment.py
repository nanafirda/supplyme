"""
Étape 6 : données complémentaires justifiées par des sources officielles (notebook §6).

- Prix   : ancrés sur les prix moyens américains du Bureau of Labor Statistics.
- Coûts  : marge brute moyenne de la distribution alimentaire (27,9 %, NGA 2025).
- Foyers : répartition officielle EIA RECS 2024, attribuée selon la taille réelle du panier.
- Moment de commande habituel : heure et jour réels d'Instacart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def tableau_ancres() -> pd.DataFrame:
    return pd.DataFrame(C.ANCRES_BLS, columns=["article BLS", "prix_usd", "période", "rayon Instacart"])


def prix_produits(cat: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    ancres = tableau_ancres()
    prix_rayon = ancres.groupby("rayon Instacart")["prix_usd"].mean()
    prix_defaut = ancres["prix_usd"].median()          # rayons sans article BLS
    cat_s = cat[cat.product_id.isin(df.product_id.unique())].copy()
    rng = np.random.default_rng(C.SEED + 1)
    ref = cat_s["department"].map(prix_rayon).fillna(prix_defaut)
    cat_s["prix"] = (ref * rng.lognormal(0, C.DISPERSION_PRIX, len(cat_s))).round(2).clip(lower=0.5)
    cat_s["cout"] = (cat_s["prix"] * (1 - C.MARGE_BRUTE)).round(2)
    cat_s["marge"] = cat_s["prix"] - cat_s["cout"]
    return cat_s


def foyers(df: pd.DataFrame) -> pd.DataFrame:
    """Clients classés par taille de panier, puis découpés selon la répartition officielle des foyers."""
    part = pd.Series(C.FOYERS_RECS) / sum(C.FOYERS_RECS.values())
    taille_panier = df.groupby(["user_id", "order_id"]).size().groupby("user_id").mean()
    rang = taille_panier.rank(method="first", pct=True)
    foyer = pd.Series(np.searchsorted(part.cumsum().values, rang.values, side="left") + 1,
                      index=rang.index).clip(upper=5)
    return pd.DataFrame({"taille_foyer": foyer, "taille_panier": taille_panier})


def habitudes_horaires(o: pd.DataFrame) -> pd.DataFrame:
    return o.groupby("user_id").agg(heure_habituelle=("order_hour_of_day", lambda s: int(s.mode().iloc[0])),
                                    jour_habituel=("order_dow", lambda s: int(s.mode().iloc[0])))
