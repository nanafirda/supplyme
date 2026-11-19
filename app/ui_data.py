"""Prépare les données RÉELLES (base SQLite) envoyées à l'interface SupplyMe (app/ui/index.html)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config as C
from database.db import metadata_get, query

FOYER_DEMO = 40294          # vrai foyer Instacart utilisé pour la démonstration mobile


def _r(df: pd.DataFrame, nd: int = 3) -> list[dict]:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(nd)
    return json.loads(out.to_json(orient="records", force_ascii=False))


def construire() -> dict:
    md = metadata_get()
    comp = query("SELECT * FROM model_comparison")
    piv = lambda t: comp[comp.task == t].pivot(index="model", columns="metric", values="value")  # noqa: E731
    reg = piv("regression").sort_values("MAE (j)").reset_index()
    cls = piv("classification").sort_values("AUC", ascending=False).reset_index()
    best_cls = md.get("meilleur_modele_classification", "")
    strat = query("SELECT * FROM strategies")

    kpis = query("""SELECT (SELECT COUNT(*) FROM customers) foyers, (SELECT COUNT(*) FROM orders) commandes,
                           (SELECT COUNT(DISTINCT product_id) FROM predictions) produits,
                           (SELECT COUNT(*) FROM predictions) predictions,
                           (SELECT COUNT(*) FROM campaigns WHERE decision != 'C — rien') recommandations,
                           (SELECT COUNT(DISTINCT customer_id) FROM campaigns WHERE decision != 'C — rien') clients_relances,
                           (SELECT COUNT(*) FROM campaigns WHERE segment = 'besoin imminent' AND decision != 'C — rien') besoins_imminents,
                           (SELECT COUNT(*) FROM campaigns WHERE segment = 'besoin dépassé' AND decision != 'C — rien') besoins_depasses
                    """).iloc[0].to_dict()
    kpis = {k: int(v) for k, v in kpis.items()}
    kpis.update(auc=float(cls.loc[cls.model == best_cls, "AUC"].iloc[0]), modele_cls=best_cls,
                mae=float(reg.iloc[0]["MAE (j)"]), modele_reg=reg.iloc[0]["model"],
                mae_base=float(reg.loc[reg.model == "Baseline", "MAE (j)"].iloc[0]),
                auc_base=float(cls.loc[cls.model == "Baseline", "AUC"].iloc[0]),
                profit=float(strat.loc[strat.strategy == "SupplyMe (ciblé)", "incremental_profit"].iloc[0]),
                date=md["date_analyse"])

    rayons = query("""SELECT p.department AS rayon, COUNT(*) AS n FROM order_items oi
                      JOIN products p ON p.product_id = oi.product_id GROUP BY rayon ORDER BY n DESC""")
    rayons["part"] = rayons.n / rayons.n.sum()
    activite = query("""SELECT predicted_date AS jour, COUNT(*) AS besoins FROM predictions
                        WHERE predicted_date > as_of_date AND predicted_date <= date(as_of_date, '+21 day')
                        GROUP BY jour ORDER BY jour""")
    envois = query("SELECT send_date AS jour, COUNT(DISTINCT customer_id) AS messages FROM campaigns WHERE decision != 'C — rien' GROUP BY jour")
    activite = activite.merge(envois, on="jour", how="left").fillna({"messages": 0})
    mois = query("SELECT strftime('%Y-%m', order_date) AS mois, COUNT(*) AS n FROM orders GROUP BY mois ORDER BY mois")
    heat = query("SELECT order_dow AS j, order_hour_of_day AS h, COUNT(*) AS n FROM orders GROUP BY j, h")

    roc = pd.read_csv(C.ARTIFACTS / "roc_curves.csv")
    roc = pd.concat([g.iloc[np.linspace(0, len(g) - 1, 30).astype(int)].assign(modele=m) for m, g in roc.groupby("modele")])
    imp = pd.read_csv(C.ARTIFACTS / "feature_importance.csv", index_col=0).head(10).reset_index(names="variable")
    from prediction.features import DESCRIPTION
    imp["description"] = imp.variable.map(DESCRIPTION)
    rvp = pd.read_csv(C.ARTIFACTS / "reel_vs_predit.csv").sample(500, random_state=0)
    val = pd.read_csv(C.ARTIFACTS / "selection_validation.csv", index_col=0).reset_index(names="modele")

    besoins = query("""
        SELECT c.customer_id AS foyer, p.product_name AS produit, p.department AS rayon, p.unit_price AS prix,
               pr.predicted_date AS date, pr.predicted_interval AS jours, pr.days_since_purchase AS depuis,
               pr.mean_interval AS cycle, c.base_proba AS proba, c.decision, c.segment, c.discount_rate AS remise,
               c.send_date AS envoi, c.send_hour AS heure, c.expected_profit AS profit
        FROM campaigns c JOIN products p ON p.product_id = c.product_id
        JOIN predictions pr ON pr.customer_id = c.customer_id AND pr.product_id = c.product_id
        ORDER BY pr.predicted_interval LIMIT 600""")
    foyers_ids = besoins.foyer.drop_duplicates().head(60).tolist()
    if FOYER_DEMO not in foyers_ids:
        foyers_ids.insert(0, FOYER_DEMO)
    ph = ",".join(map(str, foyers_ids))
    foyers = query(f"""SELECT customer_id AS foyer, household_size AS taille, ROUND(avg_basket_size, 1) AS panier,
                              preferred_hour AS heure, preferred_dow AS jour FROM customers WHERE customer_id IN ({ph})""")
    hist = query(f"""SELECT o.customer_id AS foyer, o.order_date AS date, COUNT(*) AS produits, ROUND(SUM(p.unit_price), 2) AS montant
                     FROM orders o JOIN order_items oi ON oi.order_id = o.order_id JOIN products p ON p.product_id = oi.product_id
                     WHERE o.customer_id IN ({ph}) GROUP BY o.order_id ORDER BY o.order_date""")
    habits = query(f"""SELECT pr.customer_id AS foyer, p.product_name AS produit, p.department AS rayon, pr.n_purchases AS achats,
                              ROUND(pr.mean_interval, 1) AS cycle, pr.predicted_date AS prochain, ROUND(pr.purchase_proba, 3) AS proba
                       FROM predictions pr JOIN products p ON p.product_id = pr.product_id
                       WHERE pr.customer_id IN ({ph}) AND pr.n_purchases >= 3 ORDER BY pr.n_purchases DESC""")
    habits = habits.groupby("foyer").head(12)
    top_rayon = query("""
        SELECT p.department AS rayon, p.product_name AS produit, p.unit_price AS prix,
               COUNT(DISTINCT o.customer_id) * (SUM(oi.reordered) + 1.0) / (COUNT(*) + 2.0) AS score
        FROM order_items oi JOIN orders o ON o.order_id = oi.order_id JOIN products p ON p.product_id = oi.product_id
        GROUP BY oi.product_id""")
    top_rayon = top_rayon.sort_values("score", ascending=False).groupby("rayon").head(6)
    recos = query(f"""SELECT c.customer_id AS foyer, p.product_name AS produit, p.department AS rayon, p.unit_price AS prix,
                             c.decision, c.segment, c.discount_rate AS remise, c.send_date AS envoi, c.send_hour AS heure,
                             c.base_proba AS proba, c.expected_proba AS proba_action, c.expected_profit AS profit,
                             pr.predicted_date AS date, pr.predicted_interval AS jours, pr.days_since_purchase AS depuis,
                             pr.mean_interval AS cycle
                      FROM campaigns c JOIN products p ON p.product_id = c.product_id
                      JOIN predictions pr ON pr.customer_id = c.customer_id AND pr.product_id = c.product_id
                      WHERE c.customer_id IN ({ph}) AND c.decision != 'C — rien' ORDER BY c.expected_profit DESC""")

    comptes = query("SELECT * FROM new_accounts")
    ex = pd.concat([comptes[comptes.nb_reponses == k].head(8) for k in (3, 1, 0)])
    paniers = query(f"""SELECT w.account_id, w.rank, p.product_name AS produit, p.department AS rayon, p.unit_price AS prix
                        FROM welcome_baskets w JOIN products p ON p.product_id = w.product_id
                        WHERE w.account_id IN ({",".join(repr(a) for a in ex.account_id)}) ORDER BY w.account_id, w.rank""")
    assoc = query("""SELECT a.product_id, p1.product_name AS produit, p2.product_name AS complement, a.lift, a.confidence
                     FROM product_associations a JOIN products p1 ON p1.product_id = a.product_id
                     JOIN products p2 ON p2.product_id = a.related_product_id
                     WHERE p1.product_name IN ('Banana', 'Bag of Organic Bananas', 'Organic Strawberries', 'Organic Hass Avocado',
                                               'Organic Whole Milk', 'Organic Baby Spinach', 'Large Lemon')""")
    rayons_dispo = query("SELECT DISTINCT department FROM products ORDER BY department").department.tolist()

    return {
        "kpis": kpis,
        "rayons": _r(rayons), "mois": _r(mois), "activite": _r(activite), "heat": _r(heat),
        "reg": _r(reg), "cls": _r(cls), "validation": _r(val, 4), "roc": _r(roc), "importance": _r(imp, 2),
        "rvp": _r(rvp, 1),
        "besoins": _r(besoins), "foyers": _r(foyers), "historique": _r(hist), "habitudes": _r(habits), "recos": _r(recos),
        "strategies": _r(strat, 2),
        "sensibilite": json.loads(pd.read_csv(C.ARTIFACTS / "sensibilite.csv", index_col=0).round(0).to_json(orient="split")),
        "froid_produits": _r(pd.read_csv(C.ARTIFACTS / "froid_produits.csv", index_col=0).reset_index(names="methode")),
        "froid_historique": _r(pd.read_csv(C.ARTIFACTS / "froid_historique.csv")),
        "froid_couverture": _r(pd.read_csv(C.ARTIFACTS / "froid_couverture.csv")),
        "bienvenue_eval": _r(query("SELECT * FROM new_customer_eval"), 2),
        "comptes": _r(ex), "paniers_bienvenue": _r(paniers), "associations": _r(assoc),
        "rayons_dispo": rayons_dispo, "top_rayon": _r(top_rayon, 2), "foyer_demo": FOYER_DEMO,
        "hypotheses": {"u": C.U_RELANCE, "elasticite": C.ELASTICITE, "cout_message": C.COUT_MESSAGE,
                       "points_rayon": C.POINTS_PAR_REPONSE, "bonus": C.BONUS_QUESTIONNAIRE_COMPLET, "valeur_point": C.VALEUR_POINT},
    }


EN = {
    "SupplyMe (ciblé)": "SupplyMe (targeted)", "Aucune relance": "No campaign",
    "Promo -10% à tous": "10% discount to everyone", "Relance à tous, sans réduction": "Reminder to everyone, no discount",
    "jours depuis le dernier achat du produit": "days since last purchase of the product",
    "jours depuis la dernière commande du client": "days since the customer's last order",
    "jours moyens entre deux commandes du client": "average days between the customer's orders",
    "jours depuis l'achat ÷ intervalle moyen (> 1 : en retard)": "days since purchase ÷ average interval (> 1: overdue)",
    "jours depuis la dernière commande ÷ écart moyen": "days since last order ÷ average gap",
    "part des 5 dernières commandes contenant le produit": "share of the last 5 orders containing the product",
    "nombre d'achats du produit": "number of purchases of the product",
    "part des commandes du client contenant le produit": "share of the customer's orders containing the product",
    "part de réachats du client": "customer's reorder rate", "dernier intervalle observé": "last observed interval",
    "21 et +": "21+",
    "0 commande : les 10 best-sellers": "0 orders: the 10 best-sellers", "1 commande : le 1er panier": "1 order: the 1st basket",
    "Règle : rythme des autres clients": "Rule: other customers' pace", "Modèle à froid (LightGBM)": "Cold-start model (LightGBM)",
    "Les 10 produits les plus achetés": "Top 10 most purchased products",
    "Panier de bienvenue (achetés et rachetés)": "Welcome basket (bought and rebought)",
    "Panier réparti par rayon, sans réponse": "Basket split by department, no answers",
    "Panier réparti par rayon, avec questionnaire (simulé)": "Basket split by department, with questionnaire (simulated)",
    "Placebo : réponses d'un autre client": "Placebo: another customer's answers",
    "Régression logistique": "Logistic regression", "Régression linéaire": "Linear regression",
    "LightGBM optimisé": "Tuned LightGBM",
}
CHAMPS_TRADUITS = {"strategy", "description", "historique", "situation", "methode", "method", "model", "modele",
                   "modele_cls", "modele_reg"}


def _traduire(x):
    if isinstance(x, dict):
        return {k: (EN.get(v, v) if k in CHAMPS_TRADUITS and isinstance(v, str) else _traduire(v)) for k, v in x.items()}
    if isinstance(x, list):
        return [_traduire(v) for v in x]
    return x


def page_html(langue: str = "en") -> str:
    """Interface complète : 'en' (anglais, par défaut) ou 'fr'."""
    fichier = "index_en.html" if langue == "en" else "index.html"
    gabarit = (C.ROOT / "app" / "ui" / fichier).read_text(encoding="utf-8")
    donnees = construire()
    if langue == "en":
        donnees = _traduire(donnees)
    return gabarit.replace("/*__DATA__*/{}", json.dumps(donnees, ensure_ascii=False))
