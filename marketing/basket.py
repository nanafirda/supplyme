"""Volet 6 : panier personnalisé et aperçu du message (notebook §16). Aucun envoi réel."""
from __future__ import annotations

import pandas as pd

import config as C
from database.db import query

SQL = """
SELECT c.customer_id, c.product_id, p.product_name, p.department, c.segment, c.decision, c.discount_rate,
       p.unit_price, ROUND(p.unit_price * (1 - c.discount_rate), 2) AS final_price, c.expected_profit,
       c.base_proba, c.send_date, c.send_hour, pr.predicted_date
FROM campaigns c
JOIN products p     ON p.product_id = c.product_id
JOIN predictions pr ON pr.customer_id = c.customer_id AND pr.product_id = c.product_id
WHERE c.decision != 'C — rien' AND c.customer_id = ?
ORDER BY c.expected_profit DESC
"""


def panier(client: int, db_path=C.DB_PATH) -> pd.DataFrame:
    return query(SQL, (int(client),), db_path=db_path)


def message(b: pd.DataFrame) -> str:
    if b.empty:
        return "Aucun message prévu pour ce client."
    lignes = "\n".join(f"  • {r.product_name} — {r.final_price:.2f} $" + (f"  (-{r.discount_rate:.0%})" if r.discount_rate else "")
                       for r in b.itertuples())
    economie = (b.unit_price - b.final_price).sum()
    bonus = f"\nVous économisez {economie:.2f} $ grâce à vos offres personnalisées." if economie > 0 else ""
    return (f"Envoi prévu le {pd.Timestamp(b.send_date.min()):%d/%m/%Y} à {int(b.send_hour.iloc[0])} h\n\n"
            "Objet : Votre panier habituel est prêt\n\n"
            "Bonjour, d'après vos habitudes, ces produits devraient bientôt vous manquer :\n"
            f"{lignes}\n\nTotal : {b.final_price.sum():.2f} ${bonus}\n\nValidez en un clic. — L'équipe SupplyMe")
