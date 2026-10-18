"""Part A: assert the reference checkpoints on the clean data and the database."""
import sqlite3
import pandas as pd

c = pd.read_csv("data/clean/customers.csv")
o = pd.read_csv("data/clean/orders.csv")
con = sqlite3.connect("data/instamart.db")
tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
db_orders = pd.read_sql_query("SELECT * FROM orders", con)
con.close()

comp = o[o["status"] == "completed"]
canc = o[o["status"] == "cancelled"]

checks = [
    ("12,309 order rows", len(o) == 12309),
    ("one row per order_id", o["order_id"].is_unique),
    ("11,110 completed", len(comp) == 11110),
    ("1,199 cancelled", len(canc) == 1199),
    ("no pending rows", (o["status"] == "pending").sum() == 0),
    ("0 missing order_amount", o["order_amount"].isna().sum() == 0),
    ("0 missing delivery_minutes among completed", comp["delivery_minutes"].isna().sum() == 0),
    ("1,199 missing delivery_minutes among cancelled", canc["delivery_minutes"].isna().sum() == 1199),
    ("completed revenue 2,998,668.80", round(comp["order_amount"].sum(), 2) == 2998668.80),
    ("six cities", c["city"].nunique() == 6),
    ("city counts", c["city"].value_counts().to_dict() == {"Pune": 227, "Delhi": 210, "Bengaluru": 197,
                                                           "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}),
    ("gender counts", c["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}),
    ("0 missing ages", c["age"].isna().sum() == 0),
    ("ages between 18 and 60", c["age"].between(18, 60).all()),
    ("db has three tables", {"customers", "products", "orders"} <= tables),
    ("db orders match csv", len(db_orders) == len(o)),
]
failed = [name for name, ok in checks if not ok]
for name, ok in checks:
    print(("PASS  " if ok else "FAIL  ") + name)
assert not failed, f"failed checkpoints: {failed}"
print("ALL CHECKPOINTS PASSED")
