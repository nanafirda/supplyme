"""Rapport automatique : reports/synthese.md et figures PNG réutilisables dans le rapport écrit."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

import config as C  # noqa: E402
from database.db import metadata_get, query  # noqa: E402

BLUE, ORANGE, GREY = "#1F5FBF", "#C2410C", "#9AA3AD"


def _save(fig, nom):
    (C.REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(C.REPORTS / "figures" / nom, dpi=150)
    plt.close(fig)


def figures() -> None:
    comp = query("SELECT * FROM model_comparison")
    reg = comp[(comp.task == "regression") & (comp.metric == "MAE (j)")].sort_values("value", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(reg.model, reg.value, color=[BLUE if v == reg.value.min() else GREY for v in reg.value])
    ax.set(title="Erreur moyenne (MAE, jours) — période de test", xlabel="jours")
    _save(fig, "01_comparaison_regression.png")

    roc = pd.read_csv(C.ARTIFACTS / "roc_curves.csv")
    auc = comp[(comp.task == "classification") & (comp.metric == "AUC")].set_index("model")["value"]
    meilleur = metadata_get().get("meilleur_modele_classification", "")
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for m, g in roc.groupby("modele"):
        ax.plot(g.fpr, g.tpr, lw=2.5 if m == meilleur else 1.2, label=f"{m} ({auc[m]:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=GREY)
    ax.set(title="Courbes ROC — achat sous 7 jours", xlabel="faux positifs", ylabel="vrais positifs")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, "02_courbes_roc.png")

    imp = pd.read_csv(C.ARTIFACTS / "feature_importance.csv", index_col=0)["importance"].head(12)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp.index, imp.values, color=BLUE)
    ax.set(title="Variables les plus utiles (importance, %)")
    _save(fig, "03_importance_variables.png")

    st = query("SELECT * FROM strategies")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(st.strategy[::-1], st.incremental_profit[::-1], color=[BLUE, GREY, GREY, GREY])
    ax.axvline(0, color="black", lw=0.8)
    ax.set(title="Profit incrémental estimé par stratégie ($)")
    _save(fig, "04_strategies.png")

    sens = pd.read_csv(C.ARTIFACTS / "sensibilite.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.heatmap(sens, annot=True, fmt=".0f", cmap="Blues", ax=ax)
    ax.set(title="Profit SupplyMe ($) selon les hypothèses", ylabel="effet d'une relance", xlabel="élasticité-prix")
    _save(fig, "05_sensibilite.png")


def ecrire_rapport() -> None:
    figures()
    md = metadata_get()
    comp = query("SELECT * FROM model_comparison")
    tab = lambda t: comp[comp.task == t].pivot(index="model", columns="metric", values="value").round(3)  # noqa: E731
    reg, cls = tab("regression").sort_values("MAE (j)"), tab("classification").sort_values("AUC", ascending=False)
    dec = query("SELECT segment, decision, COUNT(*) AS produits, ROUND(SUM(expected_profit), 2) AS profit "
                "FROM campaigns GROUP BY segment, decision")
    st = query("SELECT * FROM strategies").round(2)
    lignes = [
        "# SupplyMe — synthèse des résultats", "",
        f"Données Instacart : {int(md['donnees_clients_echantillon']):,} clients, "
        f"{int(md['donnees_commandes_echantillon']):,} commandes, période {md['donnees_date_min']} → {md['donnees_date_max']}.", "",
        "## Prédiction du délai avant le prochain achat (test)", "", reg.to_markdown(), "",
        "## Probabilité de rachat sous 7 jours (test)", "", cls.to_markdown(), "",
        f"Taux de base : {float(md['taux_base_7j_test']):.1%}.", "",
        "## Décisions marketing", "", dec.to_markdown(index=False), "",
        "## Rentabilité des stratégies ($)", "", st.to_markdown(index=False), "",
        "## Figures", "", *[f"![{f.name}](figures/{f.name})" for f in sorted((C.REPORTS / "figures").glob("*.png"))],
    ]
    (C.REPORTS / "synthese.md").write_text("\n".join(lignes), encoding="utf-8")
