"""Part B (tasks 4-8): lapse problem, leakage-free preprocessing, tuning, balance experiment, persistence."""
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

pd.set_option("display.width", 200)
CUTOFF = pd.Timestamp("2025-09-01")
END = pd.Timestamp("2025-10-01")

o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
c = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
p = pd.read_csv("data/clean/products.csv")
o = o.merge(p[["product_id", "category"]], on="product_id")

# ---- label uses the September window; features use ONLY orders before the cutoff ----
pre = o[o["order_ts"] < CUTOFF]
pre_comp = pre[pre["status"] == "completed"]
sept_active = set(o[(o["status"] == "completed") & (o["order_ts"] >= CUTOFF) & (o["order_ts"] < END)]["customer_id"])

g = pre_comp.groupby("customer_id")
feat = pd.DataFrame({
    "recency": (CUTOFF - g["order_ts"].max()).dt.days,
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum(),
    "avg_discount": g["discount_pct"].mean(),
    "avg_delivery": g["delivery_minutes"].mean(),
    "fav_category": g["category"].agg(lambda s: sorted(s.value_counts().sort_index().index[
        s.value_counts().sort_index().values == s.value_counts().max()])[0]),
})
final = pre[pre["status"].isin(["completed", "cancelled"])]
canc = final.groupby("customer_id")["status"].apply(lambda s: (s == "cancelled").mean())
feat["cancel_rate"] = canc.reindex(feat.index).fillna(0.0)
feat = feat.reset_index().merge(c[["customer_id", "city", "age", "gender", "signup_date"]], on="customer_id")
feat["tenure_days"] = (CUTOFF - feat["signup_date"]).dt.days
feat["lapsed"] = (~feat["customer_id"].isin(sept_active)).astype(int)
feat = feat.drop(columns="signup_date")

NUM = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery", "age", "tenure_days"]
NOM = ["city", "gender", "fav_category"]
y = feat["lapsed"]
X = feat[NUM + NOM]
print(f"cohort: {len(feat)}  lapsed: {int(y.sum())} ({y.mean()*100:.2f}%)")
assert X.isna().sum().sum() == 0, "feature matrix has missing values"
print("no missing values in any feature column: confirmed (0 NaN)")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("train/test:", len(X_tr), len(X_te))


def metrics(y_true, pred):
    return {"precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0)}


base = np.zeros(len(y_te), dtype=int)
print(f"baseline always 'not lapsed': accuracy {accuracy_score(y_te, base)*100:.1f}%  "
      f"lapsed F1 {f1_score(y_te, base, zero_division=0):.2f}")


def make_pre():
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), NOM)])


lr_pipe = Pipeline([("pre", make_pre()), ("clf", LogisticRegression(max_iter=1000))])
rf_pipe = Pipeline([("pre", make_pre()), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])

# leakage check: scaler learned on train only
tmp = clone(lr_pipe).fit(X_tr, y_tr)
sc = tmp.named_steps["pre"].named_transformers_["num"]
fi = NUM.index("frequency")
print(f"scaler mean of frequency (train only): {sc.mean_[fi]:.3f} | full-cohort mean: {X['frequency'].mean():.3f}")

gs_lr = GridSearchCV(lr_pipe, {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(X_tr, y_tr)
gs_rf = GridSearchCV(rf_pipe, {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]},
                     cv=5, scoring="f1").fit(X_tr, y_tr)
print("\nLR best:", gs_lr.best_params_, "cv F1 %.3f" % gs_lr.best_score_)
print("RF best:", gs_rf.best_params_, "cv F1 %.3f" % gs_rf.best_score_)

results = {}
for name, gs in (("LogReg", gs_lr), ("RandomForest", gs_rf)):
    pred = gs.best_estimator_.predict(X_te)
    results[name] = metrics(y_te, pred)
    print(f"\n{name} test confusion [[TN FP][FN TP]]:\n{confusion_matrix(y_te, pred)}")
    print({k: round(v, 3) for k, v in results[name].items()})

best_name = max(results, key=lambda n: results[n]["f1"])
best_gs = gs_lr if best_name == "LogReg" else gs_rf
print("\nselected model:", best_name, "test F1 %.3f" % results[best_name]["f1"])
assert results[best_name]["f1"] >= 0.55

# ---- class-balance experiment (train only, then refit the best logistic regression) ----
rs = np.random.RandomState(42)
tr = X_tr.copy(); tr["lapsed"] = y_tr.values
pos, neg = tr[tr["lapsed"] == 1], tr[tr["lapsed"] == 0]
extra = pos.sample(n=len(neg) - len(pos), replace=True, random_state=42)
bal = pd.concat([tr, extra])
print(f"\nbalanced train: lapsed {int(bal['lapsed'].sum())}, not lapsed {int((bal['lapsed']==0).sum())}")
lr_bal = clone(gs_lr.best_estimator_).fit(bal[NUM + NOM], bal["lapsed"])
m_unb = results["LogReg"]
m_bal = metrics(y_te, lr_bal.predict(X_te))
print("LogReg unbalanced:", {k: round(v, 3) for k, v in m_unb.items()})
print("LogReg balanced  :", {k: round(v, 3) for k, v in m_bal.items()})

# ---- interpretation ----
rf_best = gs_rf.best_estimator_
names = rf_best.named_steps["pre"].get_feature_names_out()
imp = pd.Series(rf_best.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
print("\nTop 5 RF features:\n", imp.head(5).round(3).to_string())
lr_best = gs_lr.best_estimator_
coef = pd.Series(lr_best.named_steps["clf"].coef_[0], index=lr_best.named_steps["pre"].get_feature_names_out())
for f in ("num__recency", "num__frequency", "num__monetary"):
    print(f"LR coef {f}: {coef[f]:+.3f}")

# ---- persist + reload ----
model = best_gs.best_estimator_
joblib.dump(model, "outputs/lapse_model.joblib")
re = joblib.load("outputs/lapse_model.joblib")
same = bool((re.predict(X_te) == model.predict(X_te)).all())
print("reloaded model reproduces test predictions exactly:", same)
assert same

# ---- retention list (illustrative: Premium flag is a whole-window label) ----
seg = pd.read_csv("outputs/customer_segments.csv")
prem = set(seg.loc[seg["is_premium"], "customer_id"])
te = feat.loc[X_te.index, ["customer_id", "lapsed"]].copy()
te["lapse_probability"] = model.predict_proba(X_te)[:, 1]
te = te[te["customer_id"].isin(prem)].sort_values("lapse_probability", ascending=False).head(10)
te = te.rename(columns={"lapsed": "actually_lapsed"})[["customer_id", "lapse_probability", "actually_lapsed"]]
te["lapse_probability"] = te["lapse_probability"].round(4)
te.to_csv("outputs/retention_list.csv", index=False)
print(f"\nretention list rows: {len(te)}; truly lapsed: {int(te['actually_lapsed'].sum())}")
print(te.to_string(index=False))
