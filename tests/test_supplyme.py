"""Tests : séparation temporelle, absence de fuite, économie des décisions, données complémentaires."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as C  # noqa: E402
from data.enrichment import foyers  # noqa: E402
from marketing import economics as eco  # noqa: E402
from marketing.decision import optimiser  # noqa: E402
from prediction.features import construire_variables, photos  # noqa: E402


def _jouet():
    lignes = []
    for k, jour in enumerate(range(0, 200, 10)):
        lignes.append((1, 100, k + 1, jour, 1, int(k > 0), 4, "produce", "Banana", 1000 + k))
    ev = pd.DataFrame(lignes, columns=["user_id", "product_id", "order_number", "jour", "add_to_cart_order",
                                       "reordered", "department_id", "department", "product_name", "order_id"])
    ev["intervalle"] = ev.groupby(["user_id", "product_id"])["jour"].diff()
    o = ev[["user_id", "order_id", "order_number", "jour"]].copy()
    o["days_since_prior_order"] = o["jour"].diff()
    clients = pd.DataFrame({"taille_foyer": [2]}, index=[1])
    return ev, o, clients


def test_aucune_information_du_futur():
    ev, o, clients = _jouet()
    f = construire_variables(95, ev, o, clients).iloc[0]
    assert f["dernier"] <= 95 and f["n_achats"] == 10           # achats aux jours 0, 10, …, 90
    assert f["intervalle_moyen"] == 10
    assert f["jours_avant_prochain_achat"] == 5                 # prochain achat au jour 100
    assert f["achat_sous_7j"] == 1


def test_separation_temporelle():
    train, test = photos()
    assert max(train) + C.HORIZON <= min(test)                  # les réponses du train ne débordent pas sur le test
    assert max(test) + C.HORIZON <= 365


def test_foyers_suivent_la_repartition_officielle():
    df = pd.DataFrame({"user_id": np.repeat(np.arange(1000), 3), "order_id": np.arange(3000)})
    df = df.loc[df.index.repeat(np.random.default_rng(0).integers(1, 20, 3000))]
    part = foyers(df)["taille_foyer"].value_counts(normalize=True).sort_index()
    attendu = pd.Series(C.FOYERS_RECS) / sum(C.FOYERS_RECS.values())
    assert np.allclose(part.values, attendu.values, atol=0.01)


def test_promo_inutile_pour_un_client_fidele():
    assert eco.gain_B(0.8, 2.79, 10, 0.05) < 0 < eco.gain_A(0.8, 2.79)


def test_trois_decisions():
    D = pd.DataFrame({
        "user_id": [1, 2, 3], "product_id": [10, 10, 10],
        "segment": ["besoin imminent", "besoin dépassé", "besoin imminent"],
        "proba_7j": [0.5, 0.2, 0.99], "valeur": [2.79, 2.79 * 6, 0.01], "prix": [10.0, 10.0, 1.0],
        "date_prevue": [C.END_DATE + pd.Timedelta(days=3)] * 3, "heure_habituelle": [18, 12, 9]})
    out = optimiser(D).set_index("user_id")
    assert out.loc[1, "decision"] == "A — relance"
    assert out.loc[2, "decision"] == "B — promo" and out.loc[2, "remise"] > 0
    assert out.loc[3, "decision"] == "C — rien" and out.loc[3, "profit"] == 0
    assert out.loc[1, "heure_envoi"] == 17
