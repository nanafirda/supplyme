"""
Étapes 1, 4 et 5 : import des fichiers Instacart, échantillon de clients,
fusion des tables et reconstruction des dates (notebook §1, §4, §5).
"""
from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

log = logging.getLogger(__name__)
OP_DTYPES = {"order_id": "int32", "product_id": "int32", "add_to_cart_order": "int16", "reordered": "int8"}


@dataclass
class Donnees:
    orders: pd.DataFrame      # commandes de l'échantillon, avec jour et date reconstruits
    df: pd.DataFrame          # table fusionnée : une ligne = un produit dans une commande
    cat: pd.DataFrame         # catalogue complet (produit, rayon fin, rayon)
    clients: np.ndarray       # identifiants des clients de l'échantillon
    qualite: dict


def decompresser(raw_dir: Path = C.RAW_DIR) -> None:
    """Décompresse les .zip éventuels (les fichiers Kaggle sont souvent livrés en .csv.zip)."""
    for z in raw_dir.glob("*.zip"):
        with zipfile.ZipFile(z) as f:
            for name in f.namelist():
                cible = raw_dir / Path(name).name
                if name.endswith(".csv") and not name.startswith("__MACOSX") and not cible.exists():
                    cible.write_bytes(f.read(name))


def fichiers_presents(raw_dir: Path = C.RAW_DIR) -> list[str]:
    decompresser(raw_dir)
    return [n for n in C.INSTACART_FILES if not (raw_dir / f"{n}.csv").exists()]


def clients_groupe(base: Path = C.BASE_GROUPE) -> np.ndarray:
    a = pd.read_excel(base / "app" / "accounts.xlsx")
    return np.sort(a.loc[a.origin == "observed_customer", "user_id"].astype(int).values)


def _lire_lignes(path: Path, order_ids: set) -> pd.DataFrame:
    morceaux = [ch[ch.order_id.isin(order_ids)]
                for ch in pd.read_csv(path, dtype=OP_DTYPES, chunksize=4_000_000)]
    return pd.concat(morceaux, ignore_index=True)


def charger(raw_dir: Path = C.RAW_DIR) -> Donnees:
    manquants = fichiers_presents(raw_dir)
    if manquants:
        raise FileNotFoundError(f"Fichiers Instacart manquants dans {raw_dir} : {manquants}")

    orders = pd.read_csv(raw_dir / "orders.csv", dtype={
        "order_id": "int32", "user_id": "int32", "order_number": "int16", "order_dow": "int8",
        "order_hour_of_day": "int8", "days_since_prior_order": "float32"})
    orders_ok = orders[orders.eval_set != "test"]          # les commandes « test » n'ont pas de produits

    # Échantillon : les 3 000 vrais clients de la base du groupe
    clients = clients_groupe()
    o = orders_ok[orders_ok.user_id.isin(clients)].sort_values(["user_id", "order_number"]).copy()
    log.info("Clients de la base du groupe : %s", f"{len(clients):,}")

    # Lignes de commande de l'échantillon (lecture par blocs : 33 M de lignes au total)
    ids = set(o.order_id)
    items = pd.concat([_lire_lignes(raw_dir / "order_products__prior.csv", ids),
                       _lire_lignes(raw_dir / "order_products__train.csv", ids)], ignore_index=True)

    cat = (pd.read_csv(raw_dir / "products.csv")
           .merge(pd.read_csv(raw_dir / "aisles.csv"), on="aisle_id")
           .merge(pd.read_csv(raw_dir / "departments.csv"), on="department_id"))

    # Fusion des tables
    df = (items.merge(o[["order_id", "user_id", "order_number", "order_dow", "order_hour_of_day",
                         "days_since_prior_order"]], on="order_id")
          .merge(cat[["product_id", "product_name", "aisle", "department", "department_id"]], on="product_id"))

    # Reconstruction des dates : cumul des écarts, puis alignement sur END_DATE avec un décalage de 0 à 13 jours
    decalage = pd.Series(np.random.default_rng(C.SEED + 2).integers(0, C.DECALAGE_MAX + 1, len(clients)),
                         index=clients)
    o["jour"] = o["days_since_prior_order"].fillna(0).groupby(o["user_id"]).cumsum()
    o["jour"] = o["jour"] - o.groupby("user_id")["jour"].transform("max") + 365 - o["user_id"].map(decalage)
    o["date"] = C.END_DATE - pd.to_timedelta(365 - o["jour"], unit="D")
    df = df.merge(o[["order_id", "jour", "date"]], on="order_id")

    qualite = {
        "clients_total": int(orders.user_id.nunique()),
        "commandes_total": int(len(orders_ok)),
        "valeurs_manquantes_days_since_prior": int(orders.days_since_prior_order.isna().sum()),
        "premieres_commandes": int((orders.order_number == 1).sum()),
        "part_ecarts_plafonnes_30j": round(float((orders_ok.days_since_prior_order == 30).mean()), 4),
        "clients_echantillon": int(len(clients)),
        "source": "base du groupe (3 000 clients réels, graine 20261118)",
        "commandes_echantillon": int(len(o)),
        "lignes_echantillon": int(len(df)),
        "date_min": str(o["date"].min().date()),
        "date_max": str(o["date"].max().date()),
    }
    log.info("Données fusionnées : %s", qualite)
    return Donnees(orders=o, df=df, cat=cat, clients=clients, qualite=qualite)
