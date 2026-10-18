"""Part B (tasks 1-3): RFM table, K-means, segment profile, PCA, outputs/customer_segments.csv."""
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 200)
os.makedirs("outputs", exist_ok=True)
FINAL_K = 4
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")

o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
comp = o[o["status"] == "completed"]
g = comp.groupby("customer_id")
rfm = pd.DataFrame({
    "recency": (SNAPSHOT - g["order_ts"].max()).dt.days,
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum().round(2),
}).reset_index()
print("customers:", len(rfm))
print("recency median/min/max:", rfm.recency.median(), rfm.recency.min(), rfm.recency.max())
print("frequency mean/max:", round(rfm.frequency.mean(), 2), rfm.frequency.max())
print("monetary mean/max:", round(rfm.monetary.mean(), 2), rfm.monetary.max())

X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rows.append({"K": k, "inertia": round(km.inertia_, 1),
                 "silhouette": round(silhouette_score(X, km.labels_), 3)})
print("\nK-means table")
print(pd.DataFrame(rows).to_string(index=False))

km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_
prof = rfm.groupby("segment_id").agg(size=("customer_id", "count"), recency=("recency", "mean"),
                                     frequency=("frequency", "mean"), monetary=("monetary", "mean")).round(2)
print("\nSegment profile (K=%d)" % FINAL_K)
print(prof.to_string())

premium_id = prof["monetary"].idxmax()
# name segments from their profile, not from the (arbitrary) cluster number
names = {}
for sid, r in prof.iterrows():
    if sid == premium_id:
        names[sid] = "Premium Big Spenders"
    elif r["frequency"] >= prof["frequency"].max() - 1e-9:
        names[sid] = "Loyal Regulars"
    elif r["recency"] >= prof["recency"].max() - 1e-9:
        names[sid] = "Lapsed / Dormant"
    else:
        names[sid] = "Occasional Shoppers"
rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"] == premium_id
print("\nsegment names:", names, "| premium segment id:", premium_id)

pca = PCA(n_components=2).fit(X)
ev = pca.explained_variance_ratio_
print("PCA explained variance:", ev.round(3), "total", round(ev.sum(), 3))

rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]] \
    .to_csv("outputs/customer_segments.csv", index=False)
print("wrote outputs/customer_segments.csv", rfm.shape)
