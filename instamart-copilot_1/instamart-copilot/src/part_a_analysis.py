"""Part A: run the seven SQL queries and the pandas cross-check / EDA."""
import re, sqlite3
import numpy as np, pandas as pd
pd.set_option("display.width", 160)
con = sqlite3.connect("data/instamart.db")
sql = open("sql/queries.sql").read()
parts = re.split(r"(?m)^-- (Q\d):.*$", sql)
Q = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}
R = {k: pd.read_sql_query(v, con) for k, v in Q.items()}

print("Q1 customers:", len(R["Q1"]))
q2 = R["Q2"]; print("Q2 rows:", len(q2), "sum:", round(q2.revenue.sum(), 2))
print("   largest cell:", q2.loc[q2.revenue.idxmax()].to_dict())
print("Q3 products:", len(R["Q3"]))
print("Q4 customers without completed order:", len(R["Q4"]))
q5 = R["Q5"]; print("Q5 customers:", len(q5), "average:", q5.avg_spend.iloc[0])
q6 = R["Q6"]
print("Q6 top per city:"); print(q6[q6.row_num == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
print("   rows where RANK != ROW_NUMBER:", int((q6.spend_rank != q6.row_num).sum()))
q7 = R["Q7"]; print("Q7 daily rows:", len(q7), "last cumulative:", q7.cumulative_revenue.iloc[-1])

# ---- pandas cross-check
o = pd.read_sql_query("SELECT * FROM orders", con); c = pd.read_sql_query("SELECT * FROM customers", con)
p = pd.read_sql_query("SELECT * FROM products", con)
comp = o[o.status == "completed"].merge(c[["customer_id", "city"]], on="customer_id").merge(p[["product_id", "category"]], on="product_id")
pt = pd.pivot_table(comp, values="order_amount", index="city", columns="category", aggfunc="sum", margins=True, fill_value=0)
print("\npivot_table grand total:", round(pt.loc["All", "All"], 2), "== SQL total:", round(q2.revenue.sum(), 2),
      "match:", abs(pt.loc["All", "All"] - q2.revenue.sum()) < 0.01)
print(pt.round(2).to_string())
agg = comp.groupby("city").agg(total=("order_amount", "sum"), mean=("order_amount", "mean"), n=("order_amount", "count")).round(2)
print("\nPer-city sum/mean/count:"); print(agg.to_string())

x = comp.order_amount
q1, q3 = x.quantile(.25), x.quantile(.75); iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"\norder_amount: mean={x.mean():.2f} median={x.median():.2f} skew={x.skew():.2f} Q1={q1:.2f} Q3={q3:.2f} "
      f"IQR={iqr:.2f} lower fence={lo:.2f} upper fence={hi:.2f} below={int((x<lo).sum())} above={int((x>hi).sum())}")
corr = comp[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print(corr.round(3).to_string())
u = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
print("largest |corr| pair:", u.abs().idxmax(), round(u[u.abs().idxmax()], 4))
