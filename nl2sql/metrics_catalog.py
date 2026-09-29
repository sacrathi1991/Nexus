"""
nl2sql/metrics_catalog.py

Step 9 of the build sequence — the semantic layer, as a literal catalog
file: a fixed list of pre-defined, pre-vetted metrics. Each one's SQL was
written and reviewed by a human (here: us, playing the "data engineer"
role) ONCE, in advance - never generated fresh by the LLM. The LLM's only
job (in nl2sql/semantic_layer.py) is to pick WHICH of these already-correct
queries answers a question, and extract its parameters - never to invent
new SQL logic.

Every sql_template uses SQLAlchemy bind parameters (:year, :limit, etc.)
rather than string interpolation - the actual values are always passed
separately and bound by the driver, so there is no SQL-injection surface
here even in principle, regardless of what the LLM extracts as a
parameter value.

Called from: nl2sql/semantic_layer.py

Step 10 addition (response shaping): each metric also carries a
response_template - a plain Python format-string that turns its result
row(s) into a natural sentence WITHOUT needing a second LLM call, since
the metric (and therefore its result shape) is already known in advance.
"multi_row" (default False) tells the formatter in semantic_layer.py
whether to format every returned row (e.g. a top-N list) instead of just
the first one.
"""

METRICS = [
    {
        "name": "total_revenue",
        "description": (
            "Total revenue across all orders (sum of unit_price * quantity * "
            "(1 - discount) from order_details), optionally filtered to one year."
        ),
        "sql_template": (
            "SELECT SUM(od.unit_price * od.quantity * (1 - od.discount)) AS total_revenue "
            "FROM order_details od JOIN orders o ON od.order_id = o.order_id "
            "WHERE (:year IS NULL OR EXTRACT(YEAR FROM o.order_date) = :year)"
        ),
        "parameters": [
            {
                "name": "year", "type": "int", "required": False,
                "description": "Filter to orders placed in this year, e.g. 1997. Omit (null) for an all-time total.",
            },
        ],
        # {year_display} is computed by the formatter, not present in the row/parameters directly
        "response_template": "Total revenue{year_display} was ${total_revenue:,.2f}.",
    },
    {
        "name": "top_products_by_revenue",
        "description": "The top N products ranked by total revenue generated.",
        "sql_template": (
            "SELECT p.product_name, SUM(od.unit_price * od.quantity * (1 - od.discount)) AS revenue "
            "FROM order_details od JOIN products p ON od.product_id = p.product_id "
            "GROUP BY p.product_name ORDER BY revenue DESC LIMIT :row_limit"
        ),
        "parameters": [
            {
                "name": "row_limit", "type": "int", "required": False, "default": 5,
                "description": "How many top products to return. Defaults to 5 if not specified.",
            },
        ],
        "response_template": "{rank}. {product_name} — ${revenue:,.2f}",
        "multi_row": True,
    },
    {
        "name": "top_employee_by_orders",
        "description": "The single employee who has processed the most orders.",
        "sql_template": (
            "SELECT e.first_name || ' ' || e.last_name AS employee_name, count(*) AS order_count "
            "FROM orders o JOIN employees e ON o.employee_id = e.employee_id "
            "GROUP BY e.employee_id, e.first_name, e.last_name ORDER BY order_count DESC LIMIT 1"
        ),
        "parameters": [],
        "response_template": "{employee_name} has processed the most orders, with {order_count} orders.",
    },
    {
        "name": "top_customer_by_orders",
        "description": "The single customer who has placed the most orders.",
        "sql_template": (
            "SELECT customer_id, count(*) AS order_count FROM orders "
            "GROUP BY customer_id ORDER BY order_count DESC LIMIT 1"
        ),
        "parameters": [],
        "response_template": "Customer {customer_id} has placed the most orders, with {order_count} orders.",
    },
    {
        "name": "orders_count_by_customer",
        "description": "How many orders a SPECIFIC customer has placed, given their customer code.",
        "sql_template": "SELECT count(*) AS order_count FROM orders WHERE customer_id = :customer_id",
        "parameters": [
            {
                "name": "customer_id", "type": "str", "required": True,
                "description": "The 5-character Northwind customer code, e.g. 'SAVEA' or 'ALFKI'.",
            },
        ],
        "response_template": "Customer {customer_id} has placed {order_count} orders.",
    },
]


def get_metric(name: str) -> dict | None:
    """Look up one metric's full definition by name - returns None if the
    name doesn't match anything in the catalog (e.g. the LLM hallucinated
    a metric name that isn't real)."""
    for metric in METRICS:
        if metric["name"] == name:
            return metric
    return None
