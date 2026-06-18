-- holocron demo warehouse: a mid-size retail/logistics business.
-- Four schemas, ~1.3M rows total, generated with generate_series + random().
-- Idempotent: drops and recreates the schemas each run.
--
-- Load with:  psql "$DATABASE_URL" -f demo/seed.sql

SET client_min_messages = warning;

DROP SCHEMA IF EXISTS sales, hr, inventory, finance CASCADE;
CREATE SCHEMA sales;
CREATE SCHEMA hr;
CREATE SCHEMA inventory;
CREATE SCHEMA finance;

-- ========================================================================
-- Schema: sales
-- ========================================================================

CREATE TABLE sales.categories (
    category_id serial PRIMARY KEY,
    name        text NOT NULL,
    parent_id   int REFERENCES sales.categories(category_id)
);

CREATE TABLE sales.products (
    product_id  serial PRIMARY KEY,
    sku         text NOT NULL UNIQUE,
    name        text NOT NULL,
    category_id int NOT NULL REFERENCES sales.categories(category_id),
    unit_price  numeric(10,2) NOT NULL,
    cost        numeric(10,2) NOT NULL,
    active      boolean NOT NULL DEFAULT true,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE sales.customers (
    customer_id serial PRIMARY KEY,
    first_name  text NOT NULL,
    last_name   text NOT NULL,
    email       text NOT NULL UNIQUE,
    city        text,
    country     text,
    segment     text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE sales.orders (
    order_id     serial PRIMARY KEY,
    customer_id  int NOT NULL REFERENCES sales.customers(customer_id),
    order_date   date NOT NULL,
    status       text NOT NULL,
    channel      text NOT NULL,
    total_amount numeric(12,2) NOT NULL DEFAULT 0
);

CREATE TABLE sales.order_items (
    order_item_id bigserial PRIMARY KEY,
    order_id      int NOT NULL REFERENCES sales.orders(order_id),
    product_id    int NOT NULL REFERENCES sales.products(product_id),
    quantity      int NOT NULL,
    unit_price    numeric(10,2) NOT NULL,
    discount      numeric(4,3) NOT NULL DEFAULT 0
);

-- ---- categories: 8 top-level + 32 children ----
INSERT INTO sales.categories (name, parent_id)
SELECT 'Category ' || g,
       CASE WHEN g <= 8 THEN NULL ELSE 1 + floor(random() * 8)::int END
FROM generate_series(1, 40) g;

-- ---- products: 2,000 ----
INSERT INTO sales.products (sku, name, category_id, unit_price, cost, active, created_at)
SELECT 'SKU-' || lpad(g::text, 6, '0'),
       (ARRAY['Widget','Gadget','Gizmo','Bracket','Cable','Adapter','Panel','Module',
              'Sensor','Valve','Filter','Bearing','Frame','Cover','Switch'])[1 + floor(random()*15)::int]
         || ' ' || (ARRAY['Pro','Lite','Max','Mini','Plus','X','S','HD'])[1 + floor(random()*8)::int]
         || ' ' || g,
       1 + floor(random() * 40)::int,
       round((5 + random() * 995)::numeric, 2),
       round((2 + random() * 400)::numeric, 2),
       random() < 0.95,
       now() - (random() * 730 || ' days')::interval
FROM generate_series(1, 2000) g;

-- ---- customers: 20,000 ----
INSERT INTO sales.customers (first_name, last_name, email, city, country, segment, created_at)
SELECT (ARRAY['James','Mary','John','Patricia','Robert','Jennifer','Michael','Linda','David',
              'Elizabeth','William','Barbara','Richard','Susan','Joseph','Jessica','Thomas',
              'Sarah','Charles','Karen'])[1 + floor(random()*20)::int],
       (ARRAY['Smith','Johnson','Williams','Brown','Jones','Garcia','Miller','Davis','Rodriguez',
              'Martinez','Hernandez','Lopez','Gonzalez','Wilson','Anderson','Thomas','Taylor',
              'Moore','Jackson','Martin'])[1 + floor(random()*20)::int],
       'user' || g || '@example.com',
       (ARRAY['New York','Los Angeles','Chicago','Houston','Nairobi','London','Berlin',
              'Toronto','Sydney','Mumbai'])[1 + floor(random()*10)::int],
       (ARRAY['USA','USA','USA','Kenya','UK','Germany','Canada','Australia','India','France'])[1 + floor(random()*10)::int],
       (ARRAY['consumer','smb','enterprise'])[1 + floor(random()*3)::int],
       now() - (random() * 1095 || ' days')::interval
FROM generate_series(1, 20000) g;

-- ---- orders: 200,000 ----
INSERT INTO sales.orders (customer_id, order_date, status, channel)
SELECT 1 + floor(random() * 20000)::int,
       (now() - (random() * 730 || ' days')::interval)::date,
       (ARRAY['pending','paid','shipped','delivered','cancelled','refunded'])[1 + floor(random()*6)::int],
       (ARRAY['web','mobile','store','partner'])[1 + floor(random()*4)::int]
FROM generate_series(1, 200000) g;

-- ---- order_items: ~1-5 per order (~600,000) ----
INSERT INTO sales.order_items (order_id, product_id, quantity, unit_price, discount)
SELECT o.order_id,
       1 + floor(random() * 2000)::int,
       1 + floor(random() * 5)::int,
       0,  -- backfilled from products below
       round((random() * 0.3)::numeric, 3)
FROM sales.orders o
CROSS JOIN LATERAL generate_series(1, (1 + floor(random() * 5))::int) gs;

-- backfill real prices from the product catalogue, then order totals
UPDATE sales.order_items oi
SET unit_price = p.unit_price
FROM sales.products p
WHERE p.product_id = oi.product_id;

UPDATE sales.orders o
SET total_amount = t.total
FROM (
    SELECT order_id, round(sum(quantity * unit_price * (1 - discount)), 2) AS total
    FROM sales.order_items
    GROUP BY order_id
) t
WHERE t.order_id = o.order_id;

CREATE INDEX ON sales.orders (customer_id);
CREATE INDEX ON sales.orders (order_date);
CREATE INDEX ON sales.order_items (order_id);
CREATE INDEX ON sales.order_items (product_id);
CREATE INDEX ON sales.products (category_id);

-- ========================================================================
-- Schema: hr
-- ========================================================================

CREATE TABLE hr.departments (
    department_id serial PRIMARY KEY,
    name          text NOT NULL,
    location      text
);

CREATE TABLE hr.employees (
    employee_id   serial PRIMARY KEY,
    first_name    text NOT NULL,
    last_name     text NOT NULL,
    email         text NOT NULL UNIQUE,
    department_id int NOT NULL REFERENCES hr.departments(department_id),
    manager_id    int REFERENCES hr.employees(employee_id),
    hire_date     date NOT NULL,
    job_title     text NOT NULL,
    salary        numeric(10,2) NOT NULL
);

INSERT INTO hr.departments (name, location)
SELECT d, (ARRAY['HQ','Remote','Nairobi','London','Berlin'])[1 + floor(random()*5)::int]
FROM unnest(ARRAY['Engineering','Sales','Marketing','Finance','Human Resources',
                  'Operations','Support','Legal','Procurement','IT','Research','Logistics']) d;

-- 5,000 employees; the first 50 are managers (no manager themselves)
INSERT INTO hr.employees (first_name, last_name, email, department_id, manager_id, hire_date, job_title, salary)
SELECT (ARRAY['James','Mary','John','Patricia','Robert','Jennifer','Michael','Linda','David',
              'Elizabeth','William','Barbara','Richard','Susan','Joseph','Jessica'])[1 + floor(random()*16)::int],
       (ARRAY['Smith','Johnson','Williams','Brown','Jones','Garcia','Miller','Davis','Rodriguez',
              'Martinez','Wilson','Anderson','Taylor','Moore','Jackson','Martin'])[1 + floor(random()*16)::int],
       'emp' || g || '@company.com',
       1 + floor(random() * 12)::int,
       CASE WHEN g <= 50 THEN NULL ELSE 1 + floor(random() * 50)::int END,
       (now() - (random() * 3650 || ' days')::interval)::date,
       (ARRAY['Analyst','Engineer','Senior Engineer','Manager','Sales Rep','Accountant',
              'Designer','Support Agent','Director','Coordinator'])[1 + floor(random()*10)::int],
       round((40000 + random() * 160000)::numeric, 2)
FROM generate_series(1, 5000) g;

CREATE INDEX ON hr.employees (department_id);
CREATE INDEX ON hr.employees (manager_id);

-- ========================================================================
-- Schema: inventory
-- ========================================================================

CREATE TABLE inventory.warehouses (
    warehouse_id serial PRIMARY KEY,
    code         text NOT NULL UNIQUE,
    city         text NOT NULL,
    country      text NOT NULL,
    capacity     int NOT NULL
);

CREATE TABLE inventory.suppliers (
    supplier_id serial PRIMARY KEY,
    name        text NOT NULL,
    country     text NOT NULL,
    rating      numeric(2,1) NOT NULL
);

CREATE TABLE inventory.stock (
    stock_id      bigserial PRIMARY KEY,
    product_id    int NOT NULL REFERENCES sales.products(product_id),
    warehouse_id  int NOT NULL REFERENCES inventory.warehouses(warehouse_id),
    quantity      int NOT NULL,
    reorder_level int NOT NULL DEFAULT 50,
    updated_at    timestamptz NOT NULL DEFAULT now(),
    UNIQUE (product_id, warehouse_id)
);

CREATE TABLE inventory.purchase_orders (
    po_id        serial PRIMARY KEY,
    supplier_id  int NOT NULL REFERENCES inventory.suppliers(supplier_id),
    warehouse_id int NOT NULL REFERENCES inventory.warehouses(warehouse_id),
    order_date   date NOT NULL,
    status       text NOT NULL,
    total        numeric(12,2) NOT NULL
);

INSERT INTO inventory.warehouses (code, city, country, capacity)
SELECT 'WH-' || lpad(g::text, 2, '0'),
       (ARRAY['New York','Los Angeles','Chicago','Houston','Nairobi','London','Berlin','Toronto','Sydney','Mumbai'])[g],
       (ARRAY['USA','USA','USA','USA','Kenya','UK','Germany','Canada','Australia','India'])[g],
       10000 + floor(random() * 90000)::int
FROM generate_series(1, 10) g;

INSERT INTO inventory.suppliers (name, country, rating)
SELECT 'Supplier ' || g,
       (ARRAY['USA','China','Germany','India','Kenya','UK','Japan','Brazil'])[1 + floor(random()*8)::int],
       round((1 + random() * 4)::numeric, 1)
FROM generate_series(1, 500) g;

-- stock: every product in every warehouse = 20,000 rows
INSERT INTO inventory.stock (product_id, warehouse_id, quantity, reorder_level, updated_at)
SELECT p.product_id, w.warehouse_id,
       floor(random() * 1000)::int, 50,
       now() - (random() * 90 || ' days')::interval
FROM sales.products p
CROSS JOIN inventory.warehouses w;

INSERT INTO inventory.purchase_orders (supplier_id, warehouse_id, order_date, status, total)
SELECT 1 + floor(random() * 500)::int,
       1 + floor(random() * 10)::int,
       (now() - (random() * 365 || ' days')::interval)::date,
       (ARRAY['draft','sent','received','cancelled'])[1 + floor(random()*4)::int],
       round((100 + random() * 50000)::numeric, 2)
FROM generate_series(1, 15000) g;

CREATE INDEX ON inventory.stock (product_id);
CREATE INDEX ON inventory.stock (warehouse_id);
CREATE INDEX ON inventory.purchase_orders (supplier_id);

-- ========================================================================
-- Schema: finance
-- ========================================================================

CREATE TABLE finance.invoices (
    invoice_id  serial PRIMARY KEY,
    order_id    int NOT NULL REFERENCES sales.orders(order_id),
    issued_date date NOT NULL,
    due_date    date NOT NULL,
    amount      numeric(12,2) NOT NULL,
    status      text NOT NULL
);

CREATE TABLE finance.payments (
    payment_id serial PRIMARY KEY,
    invoice_id int NOT NULL REFERENCES finance.invoices(invoice_id),
    paid_date  date NOT NULL,
    amount     numeric(12,2) NOT NULL,
    method     text NOT NULL
);

-- one invoice per order
INSERT INTO finance.invoices (order_id, issued_date, due_date, amount, status)
SELECT o.order_id, o.order_date, o.order_date + 30, o.total_amount,
       CASE
           WHEN o.status IN ('paid','shipped','delivered') THEN 'paid'
           WHEN o.status = 'cancelled' THEN 'void'
           ELSE 'open'
       END
FROM sales.orders o;

-- payments for paid invoices
INSERT INTO finance.payments (invoice_id, paid_date, amount, method)
SELECT i.invoice_id,
       i.issued_date + (floor(random() * 30))::int,
       i.amount,
       (ARRAY['card','bank_transfer','paypal','cash'])[1 + floor(random()*4)::int]
FROM finance.invoices i
WHERE i.status = 'paid';

CREATE INDEX ON finance.invoices (order_id);
CREATE INDEX ON finance.invoices (status);
CREATE INDEX ON finance.payments (invoice_id);

ANALYZE;

-- ---- summary ----
SELECT 'sales.customers'    AS table, count(*) FROM sales.customers
UNION ALL SELECT 'sales.products',        count(*) FROM sales.products
UNION ALL SELECT 'sales.orders',          count(*) FROM sales.orders
UNION ALL SELECT 'sales.order_items',     count(*) FROM sales.order_items
UNION ALL SELECT 'hr.employees',          count(*) FROM hr.employees
UNION ALL SELECT 'inventory.stock',       count(*) FROM inventory.stock
UNION ALL SELECT 'inventory.purchase_orders', count(*) FROM inventory.purchase_orders
UNION ALL SELECT 'finance.invoices',      count(*) FROM finance.invoices
UNION ALL SELECT 'finance.payments',      count(*) FROM finance.payments
ORDER BY 1;
