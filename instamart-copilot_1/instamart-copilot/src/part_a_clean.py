"""Part A: audit raw exports, clean them, store CSV + SQLite."""
import os, sqlite3
import numpy as np, pandas as pd

RAW, CLEAN, DB = "data/raw", "data/clean", "data/instamart.db"
os.makedirs(CLEAN, exist_ok=True); os.makedirs("outputs", exist_ok=True)

def audit():
    cust = pd.read_csv(f"{RAW}/customers.csv", keep_default_na=False)
    prod = pd.read_csv(f"{RAW}/products.csv", keep_default_na=False)
    orders = pd.read_csv(f"{RAW}/orders.csv", keep_default_na=False)
    print("== Audit: shape, dtypes, missing (blank) counts ==")
    for name, df in [("customers", cust), ("products", prod), ("orders", orders)]:
        print(f"\n[{name}] shape={df.shape}")
        blank = (df.astype(str).apply(lambda s: s.str.strip()) == "").sum()
        for c in df.columns:
            print(f"  {c:18s} dtype={str(df[c].dtype):8s} blank={blank[c]:5d} ({blank[c]/len(df)*100:.2f}%)")
    amt = pd.to_numeric(orders["order_amount"], errors="coerce")
    dm = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    age = pd.to_numeric(cust["age"], errors="coerce")
    figs = {
        "raw orders rows": len(orders),
        "exact duplicate rows": int(orders.duplicated().sum()),
        "pending rows": int((orders.status == "pending").sum()),
        "distinct order_id": int(orders.order_id.nunique()),
        "non-numeric order_amount": int(amt.isna().sum()),
        "negative delivery_minutes": int((dm < 0).sum()),
        "blank delivery_minutes": int((orders.delivery_minutes.astype(str).str.strip() == "").sum()),
        "blank age": int(age.isna().sum()),
        "age outside 10-100": int(((age < 10) | (age > 100)).sum()),
        "distinct raw city spellings": int(cust.city.nunique()),
        "distinct non-blank raw gender spellings": int(cust.gender[cust.gender.str.strip() != ""].nunique()),
        "blank gender": int((cust.gender.str.strip() == "").sum()),
    }
    print("\n== Audit figures ==")
    for k, v in figs.items():
        print(f"  {k}: {v}")
    return figs

GENDER_MAP = {"male": "male", "m": "male", "female": "female", "f": "female"}

def clean_customers():
    c = pd.read_csv(f"{RAW}/customers.csv", keep_default_na=False)
    log = {}
    before = c.city.copy()
    c["city"] = c.city.str.strip().str.title()
    log["city values changed"] = int((before != c.city).sum())
    g0 = c.gender.copy()
    c["gender"] = c.gender.str.strip().str.lower().map(GENDER_MAP).fillna("unknown")
    log["gender values changed"] = int((g0 != c.gender).sum())
    age = pd.to_numeric(c.age, errors="coerce")
    bad = (age < 10) | (age > 100)
    log["ages outside 10-100 set to missing"] = int(bad.sum())
    age = age.mask(bad)
    log["missing ages before fill (blank + invalid)"] = int(age.isna().sum())
    c["age"] = age.fillna(age.groupby(c["city"]).transform("median")).round().astype(int)
    c["signup_date"] = pd.to_datetime(c.signup_date)
    return c, log

def clean_orders(prod):
    o = pd.read_csv(f"{RAW}/orders.csv", keep_default_na=False)
    log = {"raw rows": len(o)}
    o["order_ts"] = pd.to_datetime(o.order_ts)
    n = len(o); o = o.drop_duplicates(); log["exact duplicates removed"] = n - len(o)
    n = len(o); o = o.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last")
    log["superseded (earlier pending) records removed"] = n - len(o)
    o["order_amount"] = pd.to_numeric(o.order_amount, errors="coerce")
    miss = o.order_amount.isna()
    log["unparseable amounts filled from catalogue price"] = int(miss.sum())
    price = o.product_id.map(prod.set_index("product_id").price_inr)
    calc = (price * o.quantity * (1 - o.discount_pct / 100)).round(2)
    o["order_amount"] = o.order_amount.fillna(calc)
    dm = pd.to_numeric(o.delivery_minutes, errors="coerce")
    neg = dm < 0
    log["negative delivery_minutes set to missing"] = int(neg.sum())
    dm = dm.mask(neg)
    o["delivery_minutes"] = dm
    o["category"] = o.product_id.map(prod.set_index("product_id").category)
    comp = o.status == "completed"
    med = o.loc[comp].groupby("category")["delivery_minutes"].transform("median")
    fill = o.delivery_minutes.isna() & comp
    log["completed delivery_minutes imputed with category median"] = int(fill.sum())
    o.loc[fill, "delivery_minutes"] = med[fill]
    o = o.drop(columns="category").sort_values("order_id").reset_index(drop=True)
    return o, log

def main():
    audit()
    prod = pd.read_csv(f"{RAW}/products.csv")
    cust, clog = clean_customers()
    orders, olog = clean_orders(prod)
    print("\n== Customer cleaning log =="); [print(f"  {k}: {v}") for k, v in clog.items()]
    print("== Order cleaning log =="); [print(f"  {k}: {v}") for k, v in olog.items()]
    print("cities:", cust.city.value_counts().to_dict())
    print("genders:", cust.gender.value_counts().to_dict(), "age range", cust.age.min(), cust.age.max())
    print("orders:", len(orders), orders.status.value_counts().to_dict(),
          "completed sum", round(orders.loc[orders.status == "completed", "order_amount"].sum(), 2))
    # write CSVs
    c_out = cust.copy(); c_out["signup_date"] = c_out.signup_date.dt.strftime("%Y-%m-%d")
    o_out = orders.copy(); o_out["order_ts"] = o_out.order_ts.dt.strftime("%Y-%m-%d %H:%M:%S")
    c_out.to_csv(f"{CLEAN}/customers.csv", index=False)
    prod.to_csv(f"{CLEAN}/products.csv", index=False)
    o_out.to_csv(f"{CLEAN}/orders.csv", index=False)
    if os.path.exists(DB): os.remove(DB)
    con = sqlite3.connect(DB)
    c_out.to_sql("customers", con, index=False); prod.to_sql("products", con, index=False)
    o_out.to_sql("orders", con, index=False); con.close()
    print("wrote data/clean/*.csv and", DB)

if __name__ == "__main__":
    main()
