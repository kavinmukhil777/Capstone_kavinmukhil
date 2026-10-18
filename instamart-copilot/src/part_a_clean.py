"""Part A: audit raw data, clean customers and orders, store CSV + SQLite."""
import os
import sqlite3
import pandas as pd

RAW = os.path.join("data", "raw")
CLEAN = os.path.join("data", "clean")
DB = os.path.join("data", "instamart.db")
os.makedirs(CLEAN, exist_ok=True)


def audit(df, name):
    print(f"\n=== {name}: shape={df.shape}")
    print(df.dtypes.to_string())
    miss = pd.DataFrame({"missing": df.isna().sum(), "pct": (df.isna().mean() * 100).round(2)})
    print(miss.to_string())


customers = pd.read_csv(os.path.join(RAW, "customers.csv"))
products = pd.read_csv(os.path.join(RAW, "products.csv"))
orders = pd.read_csv(os.path.join(RAW, "orders.csv"))
for df, n in ((customers, "customers"), (products, "products"), (orders, "orders")):
    audit(df, n)

# ---- audit figures ----
amt_num = pd.to_numeric(orders["order_amount"], errors="coerce")
deliv = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
age_num = pd.to_numeric(customers["age"], errors="coerce")
raw_gender = customers["gender"].dropna()
audit_fig = {
    "orders raw rows": len(orders),
    "exact duplicate rows": int(orders.duplicated().sum()),
    "pending rows": int((orders["status"] == "pending").sum()),
    "distinct order_id": int(orders["order_id"].nunique()),
    # pandas parses the literal text "N/A" as NaN on read, so count every entry that is not a number
    # (53 "N/A" + 67 "error" = 120; none of the raw amounts is truly blank)
    "non-numeric order_amount": int(amt_num.isna().sum()),
    "negative delivery_minutes": int((deliv < 0).sum()),
    "blank delivery_minutes": int(orders["delivery_minutes"].isna().sum()),
    "blank age": int(customers["age"].isna().sum()),
    "age outside 10-100": int(((age_num < 10) | (age_num > 100)).sum()),
    "distinct raw city spellings": int(customers["city"].nunique()),
    "distinct non-blank raw gender spellings": int(raw_gender.nunique()),
    "blank gender": int(customers["gender"].isna().sum()),
}
print("\n=== AUDIT FIGURES")
for k, v in audit_fig.items():
    print(f"{k}: {v}")

# ---- clean customers ----
c = customers.copy()
c["city"] = c["city"].str.strip().str.title()
gmap = {"male": "male", "m": "male", "female": "female", "f": "female"}
c["gender"] = c["gender"].fillna("").str.strip().str.lower().map(gmap).fillna("unknown")
c["age"] = pd.to_numeric(c["age"], errors="coerce")
n_bad_age = int(((c["age"] < 10) | (c["age"] > 100)).sum())
c.loc[(c["age"] < 10) | (c["age"] > 100), "age"] = float("nan")
n_missing_age = int(c["age"].isna().sum())
c["age"] = c["age"].fillna(c.groupby("city")["age"].transform("median"))
c["age"] = c["age"].round().astype(int)
c["signup_date"] = pd.to_datetime(c["signup_date"])
print("\n=== CUSTOMERS CLEANED")
print(c["city"].value_counts().to_string())
print(c["gender"].value_counts().to_string())
print("missing ages:", int(c["age"].isna().sum()), "age range:", c["age"].min(), c["age"].max())
print(f"ages set to missing (outside 10-100): {n_bad_age}; total filled with city median: {n_missing_age}")

# ---- clean orders ----
o = orders.copy()
steps = {}
o["order_ts"] = pd.to_datetime(o["order_ts"])
n0 = len(o)
o = o.drop_duplicates()
steps["exact duplicates removed"] = n0 - len(o)
n1 = len(o)
o = o.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last")
steps["superseded earlier records removed"] = n1 - len(o)
o["order_amount"] = pd.to_numeric(o["order_amount"], errors="coerce")
price = products.set_index("product_id")["price_inr"]
calc = (o["product_id"].map(price) * o["quantity"] * (1 - o["discount_pct"] / 100)).round(2)
miss_amt = o["order_amount"].isna()
steps["order_amount filled from price"] = int(miss_amt.sum())
o.loc[miss_amt, "order_amount"] = calc[miss_amt]
o["delivery_minutes"] = pd.to_numeric(o["delivery_minutes"], errors="coerce")
neg = o["delivery_minutes"] < 0
steps["negative delivery_minutes set to missing"] = int(neg.sum())
o.loc[neg, "delivery_minutes"] = float("nan")
o["category"] = o["product_id"].map(products.set_index("product_id")["category"])
comp = o["status"] == "completed"
med = o["delivery_minutes"].where(comp).groupby(o["category"]).transform("median")
fill = comp & o["delivery_minutes"].isna()
steps["completed delivery_minutes filled with category median"] = int(fill.sum())
o.loc[fill, "delivery_minutes"] = med[fill]
o = o.drop(columns="category").sort_values("order_id").reset_index(drop=True)
print("\n=== ORDER CLEANING STEPS")
for k, v in steps.items():
    print(f"{k}: {v}")
print("rows:", len(o), o["status"].value_counts().to_dict())
print("missing amount:", int(o["order_amount"].isna().sum()))
print("missing delivery completed:", int(o.loc[o.status == 'completed', 'delivery_minutes'].isna().sum()),
      "cancelled:", int(o.loc[o.status == 'cancelled', 'delivery_minutes'].isna().sum()))
print("completed revenue:", round(o.loc[o.status == "completed", "order_amount"].sum(), 2))

# ---- store ----
c_out = c.copy(); c_out["signup_date"] = c_out["signup_date"].dt.strftime("%Y-%m-%d")
o_out = o.copy(); o_out["order_ts"] = o_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
c_out.to_csv(os.path.join(CLEAN, "customers.csv"), index=False)
products.to_csv(os.path.join(CLEAN, "products.csv"), index=False)
o_out.to_csv(os.path.join(CLEAN, "orders.csv"), index=False)
if os.path.exists(DB):
    os.remove(DB)
con = sqlite3.connect(DB)
c_out.to_sql("customers", con, index=False)
products.to_sql("products", con, index=False)
o_out.to_sql("orders", con, index=False)
con.close()
print("\nstored clean CSVs and", DB)
