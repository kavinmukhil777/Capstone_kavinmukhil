"""Part B tasks 1-3: RFM, K-means (K=2..8), segment profile/naming, PCA."""
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FINAL_K = 4

o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
comp = o[o.status == "completed"]
g = comp.groupby("customer_id")
rfm = pd.DataFrame({"recency": (SNAPSHOT - g.order_ts.max()).dt.days,
                    "frequency": g.order_id.nunique(),
                    "monetary": g.order_amount.sum().round(2)}).reset_index()
print(f"RFM customers: {len(rfm)}")
print(f"recency median={rfm.recency.median():.0f} min={rfm.recency.min()} max={rfm.recency.max()}")
print(f"frequency mean={rfm.frequency.mean():.2f} max={rfm.frequency.max()}")
print(f"monetary mean={rfm.monetary.mean():.2f} max={rfm.monetary.max():.2f}")

X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
print("\nK | inertia | silhouette")
rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rows.append((k, km.inertia_, silhouette_score(X, km.labels_)))
    print(f"{k} | {km.inertia_:8.1f} | {rows[-1][2]:.3f}")

km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_
prof = rfm.groupby("segment_id").agg(size=("customer_id", "count"), mean_recency=("recency", "mean"),
                                     mean_frequency=("frequency", "mean"), mean_monetary=("monetary", "mean")).round(2)

# Business names, assigned from the profile (cluster ids carry no order)
premium_id = prof.mean_monetary.idxmax()
rest = prof.drop(index=premium_id)
loyal_id = rest.mean_frequency.idxmax()
rest = rest.drop(index=loyal_id)
dormant_id = rest.mean_recency.idxmax()
names = {premium_id: "Premium Big Spenders", loyal_id: "Loyal Regulars", dormant_id: "Dormant / At-Risk"}
for i in prof.index:
    names.setdefault(i, "Occasional Shoppers")
prof["segment_name"] = prof.index.map(names)
print("\nSegment profile:"); print(prof.to_string())
rfm["segment_name"] = rfm.segment_id.map(names)
rfm["is_premium"] = rfm.segment_id == premium_id
rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]].to_csv(
    "outputs/customer_segments.csv", index=False)

pca = PCA(n_components=2).fit(X)
r = pca.explained_variance_ratio_
print(f"\nPCA explained_variance_ratio_: {r[0]:.3f}, {r[1]:.3f}, total {r.sum():.3f}")
print("wrote outputs/customer_segments.csv; premium segment id =", premium_id)
