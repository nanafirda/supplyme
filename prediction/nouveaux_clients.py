"""
Nouveaux clients sans aucune commande (notebook §12 ter). S'appuie sur la base du groupe :
ses 1 000 nouveaux comptes fictifs, leurs réponses au questionnaire d'accueil et les règles
d'association calculées sur les vrais paniers.

1. Panier de bienvenue      : produits les plus achetés ET les plus rachetés par les clients connus.
2. Questionnaire d'accueil  : rayons préférés (0, 1 ou 3 réponses dans la base du groupe) ;
                              chaque réponse rapporte des points de fidélité (coût compté).
3. Personnalisation         : les 10 places du panier sont réparties entre les rayons au prorata des
                              préférences lissées p_d = (1{d choisi} + α·q_d) / (k + α), α = 30 (groupe).
4. Produits complémentaires : règles d'association (support, confiance, lift).

Évaluation sur données réelles : les 1 500 clients « holdout » de la base du groupe jouent les nouveaux
clients ; seuls les clients « tuning » servent à construire les classements. On mesure la part de leur
1re commande réelle retrouvée par le panier proposé (Recall@10). Les réponses au questionnaire n'existent
pas dans Instacart : pour l'évaluation, elles sont SIMULÉES à partir des achats ultérieurs du client
(bruit + 15 % de réponses inexactes) et comparées à un placebo (réponses d'un autre client).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

QUESTIONNAIRE = [
    ("rayons", "Quels sont vos rayons préférés ? (jusqu'à 3)"),
    ("taille_foyer", "Combien de personnes vivent dans votre foyer ?"),
    ("regime", "Suivez-vous un régime particulier ?"),
    ("frequence", "À quelle fréquence faites-vous vos courses ?"),
]


def points(n_rayons: int) -> int:
    """Points de fidélité : 10 par rayon indiqué, bonus si les 3 rayons sont renseignés."""
    return n_rayons * C.POINTS_PAR_REPONSE + (C.BONUS_QUESTIONNAIRE_COMPLET if n_rayons >= 3 else 0)


def classement_bienvenue(df_connus: pd.DataFrame) -> pd.DataFrame:
    """Score d'un produit = nombre d'acheteurs × probabilité de le racheter (lissée)."""
    p = df_connus.groupby("product_id").agg(acheteurs=("user_id", "nunique"), lignes=("order_id", "size"),
                                            reachats=("reordered", "sum"), rayon=("department", "first"),
                                            produit=("product_name", "first"))
    p["taux_reachat"] = (p["reachats"] + 1) / (p["lignes"] + 2)
    p["score"] = p["acheteurs"] * p["taux_reachat"]
    return p.sort_values("score", ascending=False)


def parts_rayons(df_connus: pd.DataFrame) -> pd.Series:
    q = df_connus["department"].value_counts() + 0.5
    return q / q.sum()


def preferences_lissees(rayons_choisis, q: pd.Series, alpha: float = C.ALPHA_GROUPE) -> pd.Series:
    choix = pd.Series(0.0, index=q.index)
    choix[[r for r in rayons_choisis if r in choix.index]] = 1.0
    return (choix + alpha * q) / (len(rayons_choisis) + alpha)


def panier_personnalise(classement: pd.DataFrame, pref: pd.Series, n: int = 10) -> list[int]:
    """Les n places sont réparties entre les rayons au prorata des préférences (plus forts restes),
    puis remplies avec les meilleurs produits de chaque rayon."""
    cible = pref.reindex(classement["rayon"].unique()).fillna(0)
    cible = cible / cible.sum() * n
    places = np.floor(cible).astype(int)
    places[(cible - places).nlargest(n - places.sum()).index] += 1
    choix = []
    for rayon, k in places[places > 0].items():
        choix += list(classement.index[classement["rayon"] == rayon][:k])
    return choix


def _rappel(recos: dict, verite: pd.Series) -> float:
    return float(np.mean([len(set(recos[u]) & verite[u]) / len(verite[u]) for u in verite.index]))


def evaluer(df: pd.DataFrame, base=C.BASE_GROUPE) -> pd.DataFrame:
    """Recall@10 sur la 1re commande des clients « holdout » (jamais utilisés pour construire les classements)."""
    rng = np.random.default_rng(C.SEED + 4)
    split = pd.read_excel(base / "audit" / "customer_split.xlsx")
    nouveaux = set(split.loc[split.split == "holdout", "user_id"])
    connus, test = df[~df.user_id.isin(nouveaux)], df[df.user_id.isin(nouveaux)]
    classement, q = classement_bienvenue(connus), parts_rayons(connus)
    premiere = test[test.order_number == 1].groupby("user_id")["product_id"].apply(set)
    apres = {u: g.value_counts() for u, g in test[test.order_number > 1].groupby("user_id")["department"]}
    reponses = {}
    for u in premiere.index:
        if u in apres:
            v = apres[u] * rng.lognormal(0, C.BRUIT_REPONSES, len(apres[u]))
            choix = list(v.nlargest(3).index)
            if rng.random() < C.PART_REPONSES_FAUSSES:
                choix[-1] = rng.choice([r for r in q.index if r not in choix])
            reponses[u] = choix
    users = [u for u in premiere.index if u in reponses]
    placebo = dict(zip(users, [reponses[u] for u in rng.permutation(users)]))
    pop10 = list(connus.groupby("product_id")["user_id"].nunique().nlargest(10).index)
    methodes = {
        "Les 10 produits les plus achetés": {u: pop10 for u in users},
        "Panier de bienvenue (achetés et rachetés)": {u: list(classement.index[:10]) for u in users},
        "Panier réparti par rayon, sans réponse": {u: panier_personnalise(classement, q) for u in users},
        "Panier réparti par rayon, avec questionnaire (simulé)":
            {u: panier_personnalise(classement, preferences_lissees(reponses[u], q)) for u in users},
        "Placebo : réponses d'un autre client":
            {u: panier_personnalise(classement, preferences_lissees(placebo[u], q)) for u in users},
    }
    out = pd.DataFrame({"Recall@10 (%)": {m: _rappel(r, premiere[users]) * 100 for m, r in methodes.items()}})
    out["clients évalués"] = len(users)
    return out


def comptes_et_paniers(df: pd.DataFrame, base=C.BASE_GROUPE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Les 1 000 nouveaux comptes du groupe : points gagnés et panier de bienvenue personnalisé."""
    comptes = pd.read_excel(base / "app" / "accounts.xlsx")
    comptes = comptes[comptes.origin == "synthetic_new"][["account_id", "onboarding_state", "browsing_scenario"]].copy()
    rep = pd.read_excel(base / "app" / "onboarding_answers.xlsx")
    deps = pd.read_csv(C.RAW_DIR / "departments.csv").set_index("department_id")["department"]
    rep["rayon"] = rep.department_id.map(deps)
    choix = rep.groupby("account_id")["rayon"].apply(list)
    comptes["rayons_choisis"] = comptes.account_id.map(choix).apply(lambda v: v if isinstance(v, list) else [])
    comptes["nb_reponses"] = comptes.rayons_choisis.str.len()
    comptes["points"] = comptes.nb_reponses.apply(points)
    comptes["valeur_points"] = comptes.points * C.VALEUR_POINT
    classement, q = classement_bienvenue(df), parts_rayons(df)
    lignes = []
    for r in comptes.itertuples():
        for rang, pid in enumerate(panier_personnalise(classement, preferences_lissees(r.rayons_choisis, q)), 1):
            lignes.append((r.account_id, rang, int(pid)))
    paniers = pd.DataFrame(lignes, columns=["account_id", "rank", "product_id"])
    comptes["rayons_choisis"] = comptes.rayons_choisis.apply(", ".join)
    return comptes, paniers


def associations(base=C.BASE_GROUPE, k: int = 5) -> pd.DataFrame:
    """Règles d'association de la base du groupe (vrais co-achats), les k meilleures par produit."""
    a = pd.read_excel(base / "app" / "product_associations.xlsx")
    return a.sort_values(["product_id", "lift"], ascending=[True, False]).groupby("product_id").head(k)
