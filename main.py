"""
SupplyMe — point d'entrée. Reproduit le notebook notebooks/SupplyMe_Full_Analysis.ipynb.

    python main.py run    # pipeline complet (≈ 12 min) : données → base SQL → modèles → marketing → rapport
    python main.py app    # interface Streamlit
"""
from __future__ import annotations

import argparse
import gzip
import logging
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import config as C

log = logging.getLogger("supplyme")


def run() -> None:
    from app.report import ecrire_rapport
    from data.enrichment import foyers, habitudes_horaires, prix_produits
    from data.instacart_loader import charger
    from database import db
    from marketing.decision import decider, sensibilite, strategies
    from prediction import features as F
    from prediction import models as M

    t0 = time.time()
    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if C.DB_PATH.exists():
        C.DB_PATH.unlink()

    # 1–5. Import, fusion, dates ---------------------------------------------------------
    log.info("== 1. Données Instacart")
    d = charger()
    # 6. Données complémentaires sourcées --------------------------------------------------
    cat_s = prix_produits(d.cat, d.df)
    clients_df = foyers(d.df)
    horaires = habitudes_horaires(d.orders)
    df = d.df.merge(cat_s[["product_id", "prix", "cout", "marge"]], on="product_id")

    log.info("== 2. Base SQLite")
    db.init_db()
    db.ecrire(clients_df.join(horaires).reset_index().rename(columns={
        "user_id": "customer_id", "taille_foyer": "household_size", "taille_panier": "avg_basket_size",
        "heure_habituelle": "preferred_hour", "jour_habituel": "preferred_dow"}), "customers")
    db.ecrire(cat_s.rename(columns={"prix": "unit_price", "cout": "unit_cost"})[
        ["product_id", "product_name", "aisle", "department", "unit_price", "unit_cost"]], "products")
    db.ecrire(d.orders.rename(columns={"user_id": "customer_id", "date": "order_date"})[
        ["order_id", "customer_id", "order_number", "order_date", "order_dow", "order_hour_of_day",
         "days_since_prior_order"]], "orders")
    db.ecrire(df[["order_id", "product_id", "add_to_cart_order", "reordered"]], "order_items")
    log.info("Contrôles : %s", db.controles())

    # 7–8. Historiques et habitudes -----------------------------------------------------------
    ev = F.evenements(df)
    hab = F.habitudes(ev)
    profils = hab["profil"].value_counts(normalize=True)

    # 9. Variables et séparation temporelle -------------------------------------------------
    log.info("== 3. Variables explicatives (photos train / test)")
    snap_train, snap_test = F.photos()
    TR, TE = F.jeu(snap_train, ev, d.orders, clients_df), F.jeu(snap_test, ev, d.orders, clients_df)
    log.info("Train %s obs · test %s obs · censurées %.1f %%", f"{len(TR):,}", f"{len(TE):,}", TR.censure.mean() * 100)

    # 10–12. Comparaison des modèles ---------------------------------------------------------
    log.info("== 4. Comparaison des modèles")
    tab_reg, reg_final, best_reg, reel_vs_predit = M.comparer_regression(TR, TE)
    tab_cls, rocs, importance, taux_base, probas_test = M.comparer_classification(TR, TE)
    best_cls, tab_val = M.choisir_classifieur(TR, snap_train)
    tab_val.to_csv(C.ARTIFACTS / "selection_validation.csv")
    p_opt = probas_test[best_cls]
    log.info("Modèle retenu (validation) : %s — AUC test %.3f", best_cls, tab_cls.loc[best_cls, "AUC"])
    comp = pd.concat([tab_reg.reset_index().melt(id_vars="index").assign(task="regression"),
                      tab_cls.reset_index().melt(id_vars="index").assign(task="classification")])
    db.ecrire(comp.rename(columns={"index": "model", "variable": "metric"})[["task", "model", "metric", "value"]],
              "model_comparison")
    rocs.to_csv(C.ARTIFACTS / "roc_curves.csv", index=False)
    importance.rename("importance").to_csv(C.ARTIFACTS / "feature_importance.csv")
    reel_vs_predit.to_csv(C.ARTIFACTS / "reel_vs_predit.csv", index=False)
    log.info("Régression :\n%s", tab_reg.round(3).to_string())
    log.info("Classification :\n%s", tab_cls.round(3).to_string())

    # 12 bis. Nouveaux clients sans antécédents (démarrage à froid) ------------------------
    log.info("== 4 bis. Nouveaux clients (démarrage à froid)")
    from prediction import cold_start as CS
    couverture = CS.couverture_deuxieme_commande(df)
    froid = CS.evaluer_produits_achetes_une_fois(snap_train, snap_test, ev, d.orders, clients_df)
    selon_historique = CS.precision_selon_historique(TE, p_opt)
    couverture.to_csv(C.ARTIFACTS / "froid_couverture.csv", index=False)
    froid.to_csv(C.ARTIFACTS / "froid_produits.csv")
    selon_historique.to_csv(C.ARTIFACTS / "froid_historique.csv", index=False)
    log.info("Couverture du 2e panier :\n%s", couverture.round(3).to_string(index=False))
    log.info("Produits achetés une fois :\n%s", froid.round(3).to_string())
    log.info("AUC selon l'historique :\n%s", selon_historique.round(3).to_string(index=False))

    # 12 ter. Nouveaux clients sans aucune commande -------------------------------------------
    log.info("== 4 ter. Nouveaux clients sans historique (questionnaire, panier de bienvenue)")
    from prediction import nouveaux_clients as NC
    ev_nc = NC.evaluer(df)
    log.info("Panier de bienvenue — 1re commande retrouvée :\n%s", ev_nc.round(2).to_string())
    db.ecrire(ev_nc.reset_index().rename(columns={"index": "method", "Recall@10 (%)": "recall_at_10",
                                                  "clients évalués": "customers"}), "new_customer_eval")
    comptes, paniers = NC.comptes_et_paniers(df)
    db.ecrire(comptes, "new_accounts")
    db.ecrire(paniers, "welcome_baskets")
    assoc = NC.associations()
    db.ecrire(assoc[assoc.product_id.isin(cat_s.product_id) & assoc.related_product_id.isin(cat_s.product_id)][
        ["product_id", "related_product_id", "co_purchases", "support", "confidence", "lift"]], "product_associations")

    # 13. Prédiction à la date d'analyse -------------------------------------------------------
    log.info("== 5. Prédiction au %s", C.END_DATE.date())
    tout = pd.concat([TR, TE], ignore_index=True)
    reg_final.fit(tout.loc[~tout.censure, F.VARIABLES], tout.loc[~tout.censure, "jours_avant_prochain_achat"])
    ech_final = tout.sample(min(2 * C.ECHANTILLON_MODELES_LENTS, len(tout)), random_state=C.SEED).index \
        if best_cls == "Random Forest" else tout.index
    clf_final = M.classifieur(best_cls).fit(tout.loc[ech_final, F.VARIABLES], tout.loc[ech_final, "achat_sous_7j"])
    final = clf_final[-1] if hasattr(clf_final, "steps") else clf_final
    poids = final.feature_importances_ if hasattr(final, "feature_importances_") else np.abs(final.coef_[0])
    imp = pd.Series(poids, index=F.VARIABLES)   # régression logistique : |coefficients standardisés|
    (imp / imp.sum() * 100).sort_values(ascending=False).rename("importance").to_csv(C.ARTIFACTS / "feature_importance.csv")
    P = F.construire_variables(365, ev, d.orders, clients_df, avec_cible=False)
    P[F.VARIABLES] = P[F.VARIABLES].astype(float).fillna(-1)
    P["delai_predit"] = np.clip(reg_final.predict(P[F.VARIABLES]), 1, C.HORIZON)
    P["proba_7j"] = clf_final.predict_proba(P[F.VARIABLES])[:, 1]
    P["date_prevue"] = C.END_DATE + pd.to_timedelta(P["delai_predit"].round(), unit="D")
    P = P.merge(cat_s[["product_id", "product_name", "department", "prix", "marge"]], on="product_id")
    P = P.merge(horaires, left_on="user_id", right_index=True)
    db.ecrire(pd.DataFrame({
        "customer_id": P.user_id, "product_id": P.product_id, "as_of_date": C.END_DATE,
        "last_purchase_date": C.END_DATE - pd.to_timedelta(365 - P["dernier"], unit="D"),
        "days_since_purchase": P.jours_depuis_achat, "mean_interval": P.intervalle_moyen, "n_purchases": P.n_achats,
        "predicted_interval": P.delai_predit, "predicted_date": P.date_prevue, "purchase_proba": P.proba_7j,
        "model_name": f"{best_reg} (délai) + {best_cls} (probabilité)"}), "predictions")

    # 14–17. Marketing -------------------------------------------------------------------------
    log.info("== 6. Moteur marketing")
    D = decider(P)
    db.ecrire(pd.DataFrame({
        "customer_id": D.user_id, "product_id": D.product_id, "segment": D.segment, "decision": D.decision,
        "discount_rate": D.remise, "base_proba": D.proba_7j, "expected_proba": D.proba_attendue,
        "purchase_value": D.valeur, "gain_a": D.gain_A, "gain_b": D.gain_B, "expected_profit": D.profit,
        "send_date": D.date_envoi, "send_hour": D.heure_envoi}), "campaigns")
    T = strategies(D)
    db.ecrire(T, "strategies")
    sens = sensibilite(P)
    sens.to_csv(C.ARTIFACTS / "sensibilite.csv")
    log.info("Décisions :\n%s", pd.crosstab(D.segment, D.decision).to_string())
    log.info("Stratégies :\n%s", T.round(2).to_string(index=False))

    db.metadata_set({**{f"donnees_{k}": v for k, v in d.qualite.items()},
                     "date_analyse": C.END_DATE.date(), "photos_train": len(snap_train), "photos_test": len(snap_test),
                     "obs_train": len(TR), "obs_test": len(TE), "part_censuree_train": round(TR.censure.mean(), 4),
                     "taux_base_7j_test": round(taux_base, 4), "habitudes": len(hab),
                     "habitudes_regulieres": round(float(profils.get("très régulier", 0) + profils.get("régulier", 0)), 4),
                     "meilleur_modele_regression": best_reg, "meilleur_modele_classification": best_cls})
    ecrire_rapport()
    log.info("Terminé en %.0f s. Interface : python main.py app", time.time() - t0)


def compresser(taille_max_mo: int = 20) -> None:
    """Prépare la base pour GitHub : compressée et découpée en morceaux de moins de 25 Mo."""
    gz = C.DB_PATH.with_suffix(".db.gz")
    with open(C.DB_PATH, "rb") as src, gzip.open(gz, "wb", compresslevel=9) as dst:
        shutil.copyfileobj(src, dst)
    data, n = gz.read_bytes(), taille_max_mo * 1024 * 1024
    for old in C.DB_PATH.parent.glob(C.DB_PATH.name + ".gz.part*"):
        old.unlink()
    for i in range(0, len(data), n):
        (C.DB_PATH.parent / f"{C.DB_PATH.name}.gz.part{i // n + 1}").write_bytes(data[i:i + n])
    gz.unlink()
    log.info("Base compressée en %d morceau(x)", (len(data) - 1) // n + 1)


def main() -> None:
    p = argparse.ArgumentParser(description="SupplyMe")
    p.add_argument("command", choices=["run", "app", "compress"])
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    if a.command == "run":
        run()
    elif a.command == "compress":
        compresser()
    else:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(C.ROOT / "app" / "streamlit_app.py")])


if __name__ == "__main__":
    main()
