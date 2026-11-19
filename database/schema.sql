-- =====================================================================
-- SupplyMe — schéma relationnel (SQLite). Montants en dollars US.
-- =====================================================================
PRAGMA foreign_keys = ON;

DROP VIEW  IF EXISTS v_customer_product_habits;
DROP VIEW  IF EXISTS v_purchase_intervals;
DROP TABLE IF EXISTS campaigns;
DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS model_comparison;
DROP TABLE IF EXISTS strategies;
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS metadata;
DROP TABLE IF EXISTS new_accounts;
DROP TABLE IF EXISTS welcome_baskets;
DROP TABLE IF EXISTS product_associations;
DROP TABLE IF EXISTS new_customer_eval;

-- Foyers (taille : répartition EIA RECS 2024 ; heure et jour habituels : Instacart)
CREATE TABLE customers (
    customer_id        INTEGER PRIMARY KEY,
    household_size     INTEGER NOT NULL CHECK (household_size BETWEEN 1 AND 5),
    avg_basket_size    REAL,
    preferred_hour     INTEGER CHECK (preferred_hour BETWEEN 0 AND 23),
    preferred_dow      INTEGER CHECK (preferred_dow BETWEEN 0 AND 6)
);

-- Produits (prix ancrés sur le BLS, coût = prix × (1 − 27,9 %))
CREATE TABLE products (
    product_id    INTEGER PRIMARY KEY,
    product_name  TEXT NOT NULL,
    aisle         TEXT,
    department    TEXT,
    unit_price    REAL NOT NULL CHECK (unit_price > 0),
    unit_cost     REAL NOT NULL CHECK (unit_cost >= 0),
    CHECK (unit_cost < unit_price)
);

CREATE TABLE orders (
    order_id                INTEGER PRIMARY KEY,
    customer_id             INTEGER NOT NULL,
    order_number            INTEGER NOT NULL,
    order_date              DATE    NOT NULL,      -- reconstruite (Instacart n'a pas de dates)
    order_dow               INTEGER,
    order_hour_of_day       INTEGER,
    days_since_prior_order  REAL,                  -- NULL = première commande du client
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE order_items (
    order_id           INTEGER NOT NULL,
    product_id         INTEGER NOT NULL,
    add_to_cart_order  INTEGER,
    reordered          INTEGER,
    PRIMARY KEY (order_id, product_id),
    FOREIGN KEY (order_id)   REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE TABLE predictions (
    customer_id          INTEGER NOT NULL,
    product_id           INTEGER NOT NULL,
    as_of_date           DATE    NOT NULL,
    last_purchase_date   DATE,
    days_since_purchase  REAL,
    mean_interval        REAL,
    n_purchases          INTEGER,
    predicted_interval   REAL    NOT NULL,      -- jours avant le prochain achat (régression)
    predicted_date       DATE    NOT NULL,
    purchase_proba       REAL    NOT NULL,      -- P(achat sous 7 jours) (classification)
    model_name           TEXT    NOT NULL,
    PRIMARY KEY (customer_id, product_id)
);

CREATE TABLE campaigns (
    customer_id      INTEGER NOT NULL,
    product_id       INTEGER NOT NULL,
    segment          TEXT    NOT NULL CHECK (segment IN ('besoin imminent', 'besoin dépassé')),
    decision         TEXT    NOT NULL CHECK (decision IN ('A — relance', 'B — promo', 'C — rien')),
    discount_rate    REAL    NOT NULL DEFAULT 0,
    base_proba       REAL,       -- p0 : probabilité d'achat sans message
    expected_proba   REAL,       -- probabilité avec l'action choisie
    purchase_value   REAL,       -- valeur d'un achat gagné (marge, ou marge × achats récupérés)
    gain_a           REAL,
    gain_b           REAL,
    expected_profit  REAL,       -- profit incrémental net du coût du message ($)
    send_date        DATE,
    send_hour        INTEGER,
    PRIMARY KEY (customer_id, product_id)
);

CREATE TABLE model_comparison (
    task    TEXT NOT NULL,       -- regression | classification
    model   TEXT NOT NULL,
    metric  TEXT NOT NULL,
    value   REAL,
    PRIMARY KEY (task, model, metric)
);

CREATE TABLE strategies (
    strategy             TEXT PRIMARY KEY,
    customers_contacted  INTEGER,
    incremental_profit   REAL,
    discount_cost        REAL,
    message_cost         REAL
);

CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);

-- Nouveaux clients (base du groupe) : comptes fictifs, questionnaire, points de fidélité
CREATE TABLE new_accounts (
    account_id         TEXT PRIMARY KEY,
    onboarding_state   TEXT,
    browsing_scenario  TEXT,
    rayons_choisis     TEXT,
    nb_reponses        INTEGER,
    points             INTEGER,
    valeur_points      REAL
);
CREATE TABLE welcome_baskets (
    account_id  TEXT NOT NULL,
    rank        INTEGER NOT NULL,
    product_id  INTEGER NOT NULL,
    PRIMARY KEY (account_id, rank)
);
-- Règles d'association calculées par le groupe sur les vrais paniers
CREATE TABLE product_associations (
    product_id          INTEGER,
    related_product_id  INTEGER,
    co_purchases        INTEGER,
    support             REAL,
    confidence          REAL,
    lift                REAL
);
CREATE TABLE new_customer_eval (method TEXT PRIMARY KEY, recall_at_10 REAL, customers INTEGER);

CREATE INDEX idx_orders_customer ON orders(customer_id, order_date);
CREATE INDEX idx_items_product   ON order_items(product_id);
CREATE INDEX idx_camp_decision   ON campaigns(decision);

-- Intervalle entre deux achats du même produit (fonction de fenêtre LAG)
CREATE VIEW v_purchase_intervals AS
SELECT o.customer_id, oi.product_id, o.order_date,
       julianday(o.order_date) - julianday(
           LAG(o.order_date) OVER (PARTITION BY o.customer_id, oi.product_id ORDER BY o.order_date)
       ) AS days_since_prev
FROM order_items oi JOIN orders o ON o.order_id = oi.order_id;

-- Habitudes de consommation par couple client × produit
CREATE VIEW v_customer_product_habits AS
SELECT customer_id, product_id,
       COUNT(*)                        AS n_purchases,
       MIN(order_date)                 AS first_purchase,
       MAX(order_date)                 AS last_purchase,
       ROUND(AVG(days_since_prev), 2)  AS mean_interval
FROM v_purchase_intervals
GROUP BY customer_id, product_id;
