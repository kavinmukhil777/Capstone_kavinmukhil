"""Part A: run the seven SQL queries and the pandas cross-check / exploratory analysis."""
import re
import sqlite3
import pandas as pd

pd.set_option("display.width", 200)
con = sqlite3.connect("data/instamart.db")
text = open("sql/queries.sql", encoding="utf-8").read()
parts = re.split(r"^-- (Q\d):.*$", text, flags=re.M)
queries = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}
res = {q: pd.read_sql_query(sql, con) for q, sql in queries.items()}
con.close()

print("Q1 customers:", len(res["Q1"]))
q2 = res["Q2"]
print("Q2 rows:", len(q2), "sum:", round(q2["revenue"].sum(), 2))
print("Q2 largest:", q2.iloc[0].to_dict())
print("Q3 products:", len(res["Q3"]))
print("Q4 customers:", len(res["Q4"]))
q5 = res["Q5"]
print("Q5 customers:", len(q5), "average:", q5["avg_total"].iloc[0])
q6 = res["Q6"]
print("Q6 top per city:")
print(q6[q6["row_num"] == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
print("Q6 rows with RANK != ROW_NUMBER:", int((q6["rnk"] != q6["row_num"]).sum()))
q7 = res["Q7"]
print("Q7 rows:", len(q7), "last cumulative:", q7["cumulative_revenue"].iloc[-1])

# ---- pandas cross-check ----
o = pd.read_csv("data/clean/orders.csv")
c = pd.read_csv("data/clean/customers.csv")
p = pd.read_csv("data/clean/products.csv")
comp = o[o["status"] == "completed"].merge(c[["customer_id", "city"]], on="customer_id") \
                                    .merge(p[["product_id", "category"]], on="product_id")
pv = pd.pivot_table(comp, index="city", columns="category", values="order_amount", aggfunc="sum", margins=True)
print("\npivot grand total:", round(pv.loc["All", "All"], 2), "| equals SQL total:",
      round(pv.loc["All", "All"], 2) == round(q2["revenue"].sum(), 2))
city = comp.groupby("city").agg(total=("order_amount", "sum"), mean=("order_amount", "mean"),
                                count=("order_amount", "count")).round(2)
print(city.to_string())

# ---- exploratory statistics ----
a = comp["order_amount"]
q1, q3 = a.quantile(.25), a.quantile(.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"\nmean {a.mean():.2f} median {a.median():.2f} skew {a.skew():.2f}")
print(f"Q1 {q1:.2f} Q3 {q3:.2f} IQR {iqr:.2f} lower fence {lo:.2f} upper fence {hi:.2f}")
print("outliers below:", int((a < lo).sum()), "above:", int((a > hi).sum()))
corr = comp[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print(corr.round(3).to_string())
pairs = corr.where(~pd.DataFrame(
    [[i == j for j in range(4)] for i in range(4)], index=corr.index, columns=corr.columns)).abs().stack()
print("largest |corr| pair:", pairs.idxmax(), round(corr.loc[pairs.idxmax()], 3))
