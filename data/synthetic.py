"""
Données synthétiques SupplyMe.

Deux rôles :
1. `generate_instacart_like` : génère un jeu complet AU FORMAT INSTACART
   (orders.csv, order_products__prior/train.csv, products.csv, aisles.csv,
   departments.csv). Le pipeline fonctionne ainsi sans Kaggle, et le
   chargeur Instacart est testé exactement comme avec les vraies données.
   Les comportements sont simulés à partir d'un mécanisme explicite de
   stock : un produit est racheté quand le stock du foyer est épuisé.
2. `enrich_*` : ajoute les informations économiques absentes d'Instacart
   (prix, coûts, taille du foyer, quantités).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEPARTMENT_MARGIN, DEPARTMENT_PRICE, RANDOM_SEED
from data.catalog import CATALOG

log = logging.getLogger(__name__)

HOUSEHOLD_SIZES = np.array([1, 2, 3, 4, 5])
HOUSEHOLD_PROBS = np.array([0.30, 0.32, 0.17, 0.14, 0.07])


# ---------------------------------------------------------------------------
# 1. Génération d'un jeu synthétique au format Instacart
# ---------------------------------------------------------------------------
def _consumption_factor(household_size: int) -> float:
    """Plus le foyer est grand, plus le cycle de consommation est court (réf. = 2 pers.)."""
    return (3.0 / (household_size + 1)) ** 0.7


def generate_instacart_like(out_dir: Path, n_customers: int = 3000,
                            history_days: int = 365, seed: int = RANDOM_SEED) -> Path:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    cat = pd.DataFrame(CATALOG, columns=["product_name", "aisle", "department",
                                         "cycle_days", "unit_price", "popularity"])
    cat["product_id"] = np.arange(1, len(cat) + 1)
    departments = pd.DataFrame({"department": sorted(cat["department"].unique())})
    departments["department_id"] = np.arange(1, len(departments) + 1)
    aisles = pd.DataFrame({"aisle": sorted(cat["aisle"].unique())})
    aisles["aisle_id"] = np.arange(1, len(aisles) + 1)
    cat = cat.merge(departments, on="department").merge(aisles, on="aisle").sort_values("product_id")

    pop = cat["popularity"].to_numpy(float)
    pop /= pop.sum()
    cycles = cat["cycle_days"].to_numpy(float)
    pids = cat["product_id"].to_numpy()

    orders_rows, items_rows, households = [], [], []
    order_id = 0
    for user_id in range(1, n_customers + 1):
        hh = int(rng.choice(HOUSEHOLD_SIZES, p=HOUSEHOLD_PROBS))
        households.append((user_id, hh))
        mean_gap = rng.uniform(4, 16)                     # rythme de commande propre au client
        pref_hour = int(np.clip(rng.normal(14, 3.5), 7, 22))
        n_regular = rng.integers(5, 16)
        regular = rng.choice(len(cat), size=n_regular, replace=False, p=pop)
        # cycle propre au couple client×produit
        own_cycle = cycles[regular] * _consumption_factor(hh) * rng.lognormal(0, 0.20, n_regular)
        own_cycle = np.maximum(own_cycle, 1.5)
        qty = np.clip(np.round(rng.lognormal(np.log(1 + hh / 3), 0.3, n_regular)), 1, 4).astype(int)
        # stock initial : épuisement réparti dans le premier cycle
        depletion = rng.uniform(0, own_cycle)
        # ~20 % des produits habituels sont abandonnés en cours de route (concurrence, changement de goût)
        abandon_at = np.where(rng.random(n_regular) < 0.20,
                              rng.uniform(60, history_days, n_regular), np.inf)

        t = rng.uniform(0, 20)
        prev_t, order_number = None, 0
        while t < history_days:
            order_number += 1
            order_id += 1
            day = int(t)
            # un produit est racheté si son stock est épuisé (avec une tolérance)
            due = depletion <= t + rng.uniform(0, 2, n_regular)
            # quelques oublis / achats ailleurs
            due &= rng.random(n_regular) > 0.07
            due &= t < abandon_at
            basket = [(pids[regular[j]], qty[j]) for j in np.flatnonzero(due)]
            for j in np.flatnonzero(due):
                # nouveau stock = quantité achetée × cycle unitaire, avec bruit de consommation
                depletion[j] = t + own_cycle[j] * rng.lognormal(0, 0.15)
            # achats d'impulsion
            for k in rng.choice(len(cat), size=rng.poisson(1.2), p=pop):
                basket.append((pids[k], 1))
            if basket:
                dsp = np.nan if prev_t is None else min(30, int(day - int(prev_t)))
                orders_rows.append((order_id, user_id, order_number, int(rng.integers(0, 7)),
                                    int(np.clip(pref_hour + rng.normal(0, 1.5), 0, 23)), dsp))
                seen: dict[int, int] = {}
                for pid, q in basket:
                    seen[pid] = seen.get(pid, 0) + q
                for pos, (pid, q) in enumerate(seen.items(), start=1):
                    items_rows.append((order_id, pid, pos, q))
                prev_t = t
            else:
                order_id -= 1
                order_number -= 1
            t += max(1.0, rng.gamma(4.0, mean_gap / 4.0))

    orders = pd.DataFrame(orders_rows, columns=["order_id", "user_id", "order_number", "order_dow",
                                                "order_hour_of_day", "days_since_prior_order"])
    # la dernière commande de chaque client constitue le « train » d'Instacart
    last = orders.groupby("user_id")["order_number"].transform("max") == orders["order_number"]
    orders["eval_set"] = np.where(last, "train", "prior")
    orders = orders[["order_id", "user_id", "eval_set", "order_number", "order_dow",
                     "order_hour_of_day", "days_since_prior_order"]]

    items = pd.DataFrame(items_rows, columns=["order_id", "product_id", "add_to_cart_order", "quantity"])
    tmp = items.reset_index().merge(orders[["order_id", "user_id", "order_number"]], on="order_id")
    tmp = tmp.sort_values(["user_id", "product_id", "order_number"])
    tmp["reordered"] = (tmp.groupby(["user_id", "product_id"]).cumcount() > 0).astype(int)
    items["reordered"] = tmp.set_index("index")["reordered"].reindex(items.index).to_numpy()
    train_ids = set(orders.loc[orders["eval_set"] == "train", "order_id"])
    is_train = items["order_id"].isin(train_ids)

    orders.to_csv(out_dir / "orders.csv", index=False)
    items[~is_train].to_csv(out_dir / "order_products__prior.csv", index=False)
    items[is_train].to_csv(out_dir / "order_products__train.csv", index=False)
    cat[["product_id", "product_name", "aisle_id", "department_id"]].to_csv(out_dir / "products.csv", index=False)
    aisles[["aisle_id", "aisle"]].to_csv(out_dir / "aisles.csv", index=False)
    departments[["department_id", "department"]].to_csv(out_dir / "departments.csv", index=False)
    # fichiers spécifiques au synthétique : vérité économique et taille des foyers
    cat[["product_id", "unit_price"]].to_csv(out_dir / "product_economics.csv", index=False)
    pd.DataFrame(households, columns=["user_id", "household_size"]).to_csv(out_dir / "households.csv", index=False)

    log.info("Synthétique : %s clients, %s commandes, %s lignes produits",
             f"{n_customers:,}", f"{len(orders):,}", f"{len(items):,}")
    return out_dir


# ---------------------------------------------------------------------------
# 2. Enrichissement économique (utilisé pour Instacart ET le synthétique)
# ---------------------------------------------------------------------------
def enrich_products(products: pd.DataFrame, raw_dir: Path, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Ajoute unit_price et unit_cost. Prix réels du catalogue si disponibles, sinon simulés par rayon."""
    rng = np.random.default_rng(seed + 1)
    out = products.copy()
    eco_file = raw_dir / "product_economics.csv"
    if eco_file.exists():
        out = out.merge(pd.read_csv(eco_file), on="product_id", how="left")
    else:
        base = out["category"].map(DEPARTMENT_PRICE).fillna(3.0).to_numpy()
        out["unit_price"] = np.round(base * rng.lognormal(0, 0.35, len(out)), 2).clip(0.5)
    margin = out["category"].map(DEPARTMENT_MARGIN).fillna(0.30).to_numpy()
    margin = np.clip(margin * rng.uniform(0.8, 1.2, len(out)), 0.08, 0.6)
    out["unit_cost"] = np.round(out["unit_price"] * (1 - margin), 2)
    return out


def enrich_customers(customers: pd.DataFrame, raw_dir: Path, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Ajoute household_size (réel pour le synthétique, simulé pour Instacart)."""
    rng = np.random.default_rng(seed + 2)
    out = customers.copy()
    hh_file = raw_dir / "households.csv"
    if hh_file.exists():
        hh = pd.read_csv(hh_file).rename(columns={"user_id": "customer_id"})
        out = out.merge(hh, on="customer_id", how="left")
    else:
        out["household_size"] = rng.choice(HOUSEHOLD_SIZES, size=len(out), p=HOUSEHOLD_PROBS)
    out["household_size"] = out["household_size"].fillna(2).astype(int)
    return out


def simulate_quantities(n: int, max_quantity: int, seed: int = RANDOM_SEED) -> np.ndarray:
    """Instacart ne fournit pas les quantités : loi géométrique (majorité de 1)."""
    rng = np.random.default_rng(seed + 3)
    return np.minimum(rng.geometric(0.65, n), max_quantity)
