"""
SupplyMe — application de démonstration (Streamlit).
Lancement : streamlit run app/streamlit_app.py   (ou : python main.py app)
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

import config as C  # noqa: E402
from database.db import metadata_get, preparer_base, query  # noqa: E402
from marketing import decision as dec_mod  # noqa: E402
from marketing import economics as eco  # noqa: E402
from prediction import features as feat_mod  # noqa: E402
from prediction import models as mod_mod  # noqa: E402
from data import enrichment as enr_mod  # noqa: E402

BLUE, ORANGE, GREY = "#1F5FBF", "#C2410C", "#9AA3AD"
COULEURS = {"A — relance": BLUE, "B — promo": ORANGE, "C — rien": GREY}
GITHUB = "https://github.com/nanafirda/supplyme"

st.set_page_config(page_title="SupplyMe", page_icon=":material/shopping_basket:", layout="wide", initial_sidebar_state="collapsed")
preparer_base()
if not C.DB_PATH.exists():
    st.error("Base introuvable : lancez d'abord  `python main.py run`.")
    st.stop()


@st.cache_data(show_spinner=False)
def q(sql: str, params: tuple = ()) -> pd.DataFrame:
    return query(sql, params)


@st.cache_data(show_spinner=False)
def meta() -> dict:
    return metadata_get()


@st.cache_data(show_spinner=False)
def candidats() -> pd.DataFrame:
    return q("""
        SELECT c.customer_id AS user_id, c.product_id, c.segment, c.base_proba AS proba_7j, c.purchase_value AS valeur,
               p.unit_price AS prix, p.product_name, p.department, pr.predicted_date AS date_prevue,
               pr.mean_interval AS intervalle_moyen, pr.days_since_purchase AS jours_depuis_achat,
               cu.preferred_hour AS heure_habituelle
        FROM campaigns c
        JOIN products p     ON p.product_id = c.product_id
        JOIN predictions pr ON pr.customer_id = c.customer_id AND pr.product_id = c.product_id
        JOIN customers cu   ON cu.customer_id = c.customer_id""").assign(date_prevue=lambda d: pd.to_datetime(d.date_prevue))


def usd(x: float) -> str:
    return f"{x:,.0f} $".replace(",", " ")


md = meta()
from app.ui_data import page_html  # noqa: E402


@st.cache_data(show_spinner="Loading SupplyMe…")
def interface(langue: str) -> str:
    return page_html(langue)


langue = st.sidebar.radio("Language / Langue", ["English", "Français"])
vue = st.sidebar.radio("View / Vue", ["SupplyMe app", "Technical workshop (code)"],
                       help="The app is the final interface. The workshop shows the Python calculations live.")
if vue == "SupplyMe app":
    st.markdown("<style>.block-container{padding:0 .5rem!important;max-width:100%!important}"
                "header[data-testid='stHeader']{background:transparent}</style>", unsafe_allow_html=True)
    if hasattr(st, "iframe"):
        st.iframe(interface("en" if langue == "English" else "fr"), height=1850)
    else:  # anciennes versions de Streamlit
        import streamlit.components.v1 as components
        components.html(interface("en" if langue == "English" else "fr"), height=1850, scrolling=True)
    st.stop()

st.title("SupplyMe")
st.caption(f"Predict. Anticipate. Convert. · Données réelles Instacart · {int(md['donnees_clients_echantillon']):,} foyers · "
           f"date d'analyse : {md['date_analyse']} · montants en dollars US".replace(",", " "))


t1, t2, t3, t4, t6, t5 = st.tabs(["01 · Tableau de bord", "02 · Foyers et habitudes", "03 · Prédictions",
                                  "04 · Marketing", "05 · Nouveaux clients", "06 · Méthode et code"])

# ---------------------------------------------------------------------------------------------
with t1:
    strat = q("SELECT * FROM strategies").set_index("strategy")
    sm = strat.loc["SupplyMe (ciblé)", "incremental_profit"]
    meilleur_simple = strat.drop(["SupplyMe (ciblé)", "Aucune relance"])["incremental_profit"].max()
    auc = q("SELECT value FROM model_comparison WHERE task='classification' AND model=? AND metric='AUC'",
            (md["meilleur_modele_classification"],)).value[0]
    k = st.columns(5)
    k[0].metric("Foyers analysés", f"{int(md['donnees_clients_echantillon']):,}".replace(",", " "))
    k[1].metric("Commandes", f"{int(md['donnees_commandes_echantillon']):,}".replace(",", " "))
    k[2].metric("Habitudes détectées", f"{int(md['habitudes']):,}".replace(",", " "))
    k[3].metric("AUC du modèle", f"{auc:.3f}")
    k[4].metric("Profit SupplyMe", usd(sm), f"+{(sm / meilleur_simple - 1):.0%} vs meilleure stratégie simple")

    g1, g2 = st.columns(2)
    mois = q("SELECT strftime('%Y-%m', order_date) AS mois, COUNT(*) AS commandes FROM orders GROUP BY mois ORDER BY mois")
    g1.plotly_chart(px.bar(mois, x="mois", y="commandes", title="Commandes par mois", color_discrete_sequence=[BLUE]),
                    width="stretch")
    heat = q("SELECT order_dow AS jour, order_hour_of_day AS heure, COUNT(*) AS n FROM orders GROUP BY jour, heure")
    g2.plotly_chart(px.density_heatmap(heat, x="heure", y="jour", z="n", nbinsx=24, nbinsy=7, color_continuous_scale="Blues",
                                       title="Quand les foyers commandent (jour × heure)"), width="stretch")
    g3, g4 = st.columns(2)
    rayons = q("""SELECT p.department AS rayon, ROUND(SUM(p.unit_price)) AS ca FROM order_items oi
                  JOIN products p ON p.product_id = oi.product_id GROUP BY rayon ORDER BY ca DESC LIMIT 12""")
    g3.plotly_chart(px.bar(rayons, x="ca", y="rayon", orientation="h", title="Chiffre d'affaires par rayon ($)",
                           color_discrete_sequence=[BLUE]).update_yaxes(autorange="reversed"), width="stretch")
    decs = q("SELECT segment, decision, COUNT(*) AS produits FROM campaigns GROUP BY segment, decision")
    g4.plotly_chart(px.bar(decs, x="segment", y="produits", color="decision", color_discrete_map=COULEURS,
                           title="Décisions marketing par segment"), width="stretch")

# ---------------------------------------------------------------------------------------------
with t2:
    ids = q("SELECT DISTINCT customer_id FROM campaigns WHERE decision != 'C — rien' ORDER BY customer_id").customer_id.tolist()
    cid = st.selectbox("Foyer", ids, key="foyer")
    info = q("SELECT * FROM customers WHERE customer_id = ?", (cid,)).iloc[0]
    hist = q("""SELECT o.order_date AS date, COUNT(*) AS produits, ROUND(SUM(p.unit_price), 2) AS montant
                FROM orders o JOIN order_items oi ON oi.order_id = o.order_id JOIN products p ON p.product_id = oi.product_id
                WHERE o.customer_id = ? GROUP BY o.order_id ORDER BY o.order_date""", (cid,))
    k = st.columns(4)
    k[0].metric("Taille du foyer", f"{int(info.household_size)} pers.")
    k[1].metric("Commandes", len(hist))
    k[2].metric("Panier moyen", f"{hist.montant.mean():.2f} $")
    k[3].metric("Heure habituelle", f"{int(info.preferred_hour)} h")
    st.plotly_chart(px.line(hist, x="date", y="montant", markers=True, title="Historique des commandes ($)",
                            color_discrete_sequence=[BLUE]), width="stretch")
    st.subheader("Habitudes de consommation et prochain achat prévu")
    hab = q("""SELECT p.product_name AS produit, p.department AS rayon, pr.n_purchases AS achats,
                      ROUND(pr.mean_interval, 1) AS "intervalle habituel (j)", ROUND(pr.days_since_purchase) AS "depuis (j)",
                      pr.predicted_date AS "prochain achat prévu", ROUND(pr.purchase_proba, 3) AS "P(achat sous 7 j)"
               FROM predictions pr JOIN products p ON p.product_id = pr.product_id
               WHERE pr.customer_id = ? AND pr.n_purchases >= 3 ORDER BY pr.n_purchases DESC""", (cid,))
    st.dataframe(hab, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------------------------
with t3:
    comp = q("SELECT * FROM model_comparison")
    piv = lambda t: comp[comp.task == t].pivot(index="model", columns="metric", values="value")  # noqa: E731
    reg, cls = piv("regression").sort_values("MAE (j)"), piv("classification").sort_values("AUC", ascending=False)
    st.markdown("Tous les modèles sont évalués sur une **période future jamais vue** pendant l'entraînement (août–septembre).")
    c1, c2 = st.columns(2)
    c1.subheader("Délai avant le prochain achat")
    c1.dataframe(reg.style.format("{:.2f}"), width="stretch")
    c2.subheader("Rachat sous 7 jours")
    c2.dataframe(cls.style.format("{:.3f}", na_rep="—"), width="stretch")
    c3, c4 = st.columns(2)
    roc = pd.read_csv(C.ARTIFACTS / "roc_curves.csv")
    c3.plotly_chart(px.line(roc, x="fpr", y="tpr", color="modele", title="Courbes ROC",
                            labels={"fpr": "faux positifs", "tpr": "vrais positifs"}), width="stretch")
    imp = pd.read_csv(C.ARTIFACTS / "feature_importance.csv", index_col=0).head(12).reset_index(names="variable")
    imp["description"] = imp.variable.map(feat_mod.DESCRIPTION)
    c4.plotly_chart(px.bar(imp, x="importance", y="variable", orientation="h", hover_data=["description"],
                           title="Variables les plus utiles (%)", color_discrete_sequence=[BLUE])
                    .update_yaxes(autorange="reversed"), width="stretch")
    rv = pd.read_csv(C.ARTIFACTS / "reel_vs_predit.csv")
    st.plotly_chart(px.scatter(rv, x="reel", y="predit", opacity=0.25, title="Délai réel vs prédit (échantillon du test)",
                               labels={"reel": "réel (jours)", "predit": "prédit (jours)"}, color_discrete_sequence=[BLUE]),
                    width="stretch")

# ---------------------------------------------------------------------------------------------
with t4:
    st.info("La probabilité d'achat spontané vient du modèle. L'effet d'une réduction vient de la littérature "
            "(élasticité −2,62, Bijmolt et al. 2005). L'effet d'un simple rappel et le coût d'un message sont des "
            "hypothèses : faites-les varier.")
    s1, s2, s3 = st.columns(3)
    u = s1.slider("Effet d'un rappel (part des hésitants convaincus)", 0.0, 0.10, C.U_RELANCE, 0.005, format="%.3f")
    e = s2.slider("Élasticité-prix", 1.0, 4.0, C.ELASTICITE, 0.01)
    c = s3.slider("Coût d'un message ($)", 0.0, 0.50, C.COUT_MESSAGE, 0.01)

    @st.cache_data(show_spinner="Optimisation des campagnes…")
    def simuler(u, e, c):
        D = dec_mod.optimiser(candidats(), u, e, c)
        return D, dec_mod.strategies(D, u, e, c)

    D, T = simuler(u, e, c)
    m1, m2 = st.columns([3, 2])
    m1.plotly_chart(px.bar(T, x="incremental_profit", y="strategy", orientation="h", title="Profit incrémental par stratégie ($)",
                           color="strategy", color_discrete_sequence=[GREY, GREY, GREY, BLUE])
                    .update_layout(showlegend=False).update_yaxes(autorange="reversed"), width="stretch")
    m2.dataframe(T.rename(columns={"strategy": "stratégie", "customers_contacted": "clients", "incremental_profit": "profit ($)",
                                   "discount_cost": "réductions ($)", "message_cost": "messages ($)"}).round(0),
                 width="stretch", hide_index=True)
    st.dataframe(pd.crosstab(D.segment, D.decision, margins=True, margins_name="Total"), width="stretch")

    st.subheader("Parcours d'un client : besoin prévu → décision → panier → message")
    relances = D[D.decision != "C — rien"].groupby("user_id").size()
    cid = st.selectbox("Client relancé", relances.index.tolist(), key="client_mkt")
    b = D[(D.user_id == cid) & (D.decision != "C — rien")].sort_values("profit", ascending=False)
    st.dataframe(b[["product_name", "segment", "date_prevue", "proba_7j", "gain_A", "gain_B", "decision", "remise", "prix", "profit"]]
                 .rename(columns={"product_name": "produit", "date_prevue": "besoin prévu", "proba_7j": "p0",
                                  "gain_A": "gain relance ($)", "gain_B": "gain promo ($)", "prix": "prix ($)", "profit": "profit ($)"})
                 .round(3), width="stretch", hide_index=True)
    b = b.assign(prix_final=(b.prix * (1 - b.remise)).round(2))
    lignes = "\n".join(f"  • {r.product_name} — {r.prix_final:.2f} $" + (f"  (-{r.remise:.0%})" if r.remise else "")
                       for r in b.itertuples())
    st.code(f"Envoi : {pd.Timestamp(b.date_envoi.min()):%d/%m/%Y} à {int(b.heure_envoi.iloc[0])} h\n\n"
            f"Objet : Votre panier habituel est prêt\n\nBonjour, d'après vos habitudes, ces produits devraient bientôt vous manquer :\n"
            f"{lignes}\n\nTotal : {b.prix_final.sum():.2f} $ — validez en un clic.", language=None)

# ---------------------------------------------------------------------------------------------
with t6:
    st.markdown("Un client qui n'a encore rien acheté reçoit un **panier de bienvenue**, peut remplir un "
                "**questionnaire d'accueil** (10 points par rayon indiqué, 50 points de bonus pour 3 rayons ; 100 points = 1 $), "
                "puis voit son panier **personnalisé** selon ses réponses. Dès son 1er achat, le **modèle à froid** prend le relais.")
    st.caption("Comptes, réponses et points : données FICTIVES de la base du groupe (démonstration). "
               "Évaluation : 1re commande RÉELLE des 1 500 clients « holdout ».")
    e = q("SELECT method AS méthode, ROUND(recall_at_10, 2) AS \"1re commande retrouvée (%)\" FROM new_customer_eval")
    n1, n2 = st.columns([3, 2])
    n1.dataframe(e, hide_index=True, width="stretch")
    froid = pd.read_csv(C.ARTIFACTS / "froid_produits.csv", index_col=0)
    n2.markdown("**Après un 1er achat d'un produit (modèle à froid, données réelles)**")
    n2.dataframe(froid[["AUC (achat sous 7 j)", "MAE délai (j)"]].round(3), width="stretch")
    hist = pd.read_csv(C.ARTIFACTS / "froid_historique.csv")
    st.plotly_chart(px.bar(hist, x="historique", y="AUC", text=hist.AUC.round(3), range_y=[0.6, 0.85],
                           title="Plus le client a d'historique, plus la prédiction est fiable",
                           labels={"historique": "commandes déjà passées"}, color_discrete_sequence=[BLUE]), width="stretch")
    comptes = q("SELECT * FROM new_accounts")
    st.dataframe(comptes.groupby("nb_reponses").agg(comptes=("account_id", "size"), points=("points", "first"),
                                                    valeur=("valeur_points", "first")).reset_index()
                 .rename(columns={"nb_reponses": "rayons indiqués", "valeur": "valeur ($)"}), hide_index=True)
    acc = st.selectbox("Nouveau compte", comptes.account_id.tolist(),
                       index=int(comptes.index[comptes.nb_reponses == 3][0]), key="nouveau")
    c = comptes[comptes.account_id == acc].iloc[0]
    st.markdown(f"Rayons choisis : **{c.rayons_choisis or 'aucun'}** · points gagnés : **{c.points}** ({c.valeur_points:.2f} $)")
    panier_b = q("""SELECT w.rank AS rang, p.product_name AS produit, p.department AS rayon, p.unit_price AS "prix ($)"
                    FROM welcome_baskets w JOIN products p ON p.product_id = w.product_id
                    WHERE w.account_id = ? ORDER BY w.rank""", (acc,))
    st.dataframe(panier_b, hide_index=True, width="stretch")

# ---------------------------------------------------------------------------------------------
with t5:
    def code(obj, legende):
        st.caption(legende)
        st.code(inspect.getsource(obj), language="python")

    st.markdown(f"""
L'analyse complète, avec tous les graphiques, est dans le notebook
[`notebooks/SupplyMe_Full_Analysis.ipynb`]({GITHUB}/blob/main/notebooks/SupplyMe_Full_Analysis.ipynb).
Cette application utilise **exactement le même code** (dossiers `data/`, `prediction/`, `marketing/`).
""")
    with st.expander("① Données : import, valeurs manquantes, dates, données complémentaires", expanded=True):
        st.markdown("""
- **Base du groupe** : 3 000 vrais clients Instacart tirés au hasard (graine 20261118) + 1 000 nouveaux comptes fictifs avec questionnaire d'accueil.
- **Valeurs manquantes** : seule `days_since_prior_order` en contient, exactement les 206 209 **premières commandes** (pas de commande précédente). Ce ne sont pas des erreurs.
- **Dates** : reconstruites par cumul des écarts entre commandes, alignées sur le 31 octobre 2026.
- **Données complémentaires sourcées** : prix (BLS), marge brute 27,9 % (NGA 2025), taille des foyers (EIA RECS 2024), élasticité-prix −2,62 (Bijmolt et al. 2005).
""")
        st.dataframe(enr_mod.tableau_ancres(), hide_index=True, width="stretch")
        code(enr_mod.prix_produits, "data/enrichment.py — prix ancrés sur le BLS")
        code(enr_mod.foyers, "data/enrichment.py — taille des foyers")
    with st.expander("② Prédiction : variables, séparation temporelle, 5 modèles"):
        st.markdown("""
- **25 variables** : rythme du produit (intervalles, retard), fidélité (part des commandes avec le produit), rythme du client, popularité du produit.
- **Séparation temporelle** : 12 photos d'entraînement (janvier–juin), une pause de 60 jours, puis 3 photos de test (août–septembre). Aucune information du futur n'est utilisée.
- **Modèles comparés** : règle simple, régression linéaire/logistique, Random Forest, XGBoost, LightGBM. Le modèle de probabilité est **choisi sur une période de validation** (jamais sur le test) : avec 3 000 clients, c'est la **régression logistique** qui l'emporte ; pour le délai, LightGBM.
""")
        st.dataframe(pd.DataFrame({"variable": feat_mod.VARIABLES, "signification": [feat_mod.DESCRIPTION[v] for v in feat_mod.VARIABLES]}),
                     hide_index=True, width="stretch")
        code(feat_mod.photos, "prediction/features.py — séparation train / test dans le temps")
        code(mod_mod.comparer_classification, "prediction/models.py — comparaison des modèles")
    with st.expander("③ Marketing : relance, promo ou rien"):
        st.markdown("""
| Scénario | Profit incrémental |
|---|---|
| **A — relance** | (pA − p0) × V − c |
| **B — promo d** | pB × (V − d × prix) − p0 × V − c |
| **C — rien** | 0 |

- **p0** : probabilité d'achat spontané (modèle) ; **pA** = p0 + u(1 − p0) ; **pB** = pA × (1 + 2,62 d).
- **V** : la marge d'un achat (besoin imminent), ou la marge × les achats récupérés sur 90 jours (besoin dépassé : on reconquiert une habitude).
- **Effet d'aubaine** : la réduction est payée aussi à ceux qui auraient acheté sans elle.
- **Moment d'envoi** : 1 jour avant le besoin, 1 h avant l'heure habituelle de commande.
""")
        code(eco, "marketing/economics.py")
        code(dec_mod.optimiser, "marketing/decision.py — décision A / B / C")
    with st.expander("④ Limites"):
        st.markdown("""
- Dates reconstruites ; prix et foyers ancrés sur des sources officielles mais simulés au niveau individuel.
- Instacart plafonne les écarts entre commandes à 30 jours.
- L'effet d'un rappel et le coût d'un message sont des hypothèses : un A/B test permettrait de les mesurer.
""")
