"""Part B tasks 4-8: lapse definition, leakage-free preprocessing, tuning, balancing, interpretation, persistence."""
import joblib, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CUTOFF, END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-10-01")
o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
cust = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
prod = pd.read_csv("data/clean/products.csv")
o = o.merge(prod[["product_id", "category"]], on="product_id")

# ---- features: ONLY orders before the cutoff
pre = o[o.order_ts < CUTOFF]
comp = pre[pre.status == "completed"]
g = comp.groupby("customer_id")
fav = (comp.groupby(["customer_id", "category"]).size().rename("n").reset_index()
       .sort_values(["customer_id", "n", "category"], ascending=[True, False, True])
       .drop_duplicates("customer_id").set_index("customer_id").category)
final = pre[pre.status.isin(["completed", "cancelled"])]
canc_rate = final.groupby("customer_id").status.apply(lambda s: (s == "cancelled").mean())
F = pd.DataFrame({
    "recency": (CUTOFF - g.order_ts.max()).dt.days,
    "frequency": g.order_id.nunique(),
    "monetary": g.order_amount.sum(),
    "avg_discount": g.discount_pct.mean(),
    "cancel_rate": canc_rate,
    "avg_delivery": g.delivery_minutes.mean(),
    "fav_category": fav}).dropna(subset=["frequency"])
F = F.join(cust.set_index("customer_id")[["city", "age", "gender", "signup_date"]])
F["tenure_days"] = (CUTOFF - F.signup_date).dt.days
F = F.drop(columns="signup_date")
# ---- label: no completed order in September
sept = o[(o.status == "completed") & (o.order_ts >= CUTOFF) & (o.order_ts < END)].customer_id.unique()
y = (~F.index.isin(sept)).astype(int)
y = pd.Series(y, index=F.index, name="lapsed")
print(f"cohort={len(F)} lapsed={y.sum()} ({y.mean()*100:.2f}%)")

NUM = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery", "age", "tenure_days"]
CAT = ["city", "gender", "fav_category"]
FEATS = NUM + CAT
X = F[FEATS]
assert not X.isna().any().any(), "missing value in features"
print("no missing values in any feature column: OK")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"train={len(Xtr)} test={len(Xte)}")

def prep():
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])

def metrics(model, name):
    p = model.predict(Xte)
    cm = confusion_matrix(yte, p)
    print(f"{name}: confusion [[TN FP],[FN TP]] = {cm.tolist()}  precision={precision_score(yte,p,zero_division=0):.3f} "
          f"recall={recall_score(yte,p,zero_division=0):.3f} F1={f1_score(yte,p,zero_division=0):.3f}")
    return p

# ---- baseline
base = np.zeros(len(yte), dtype=int)
print(f"\nBaseline (always not lapsed): accuracy={accuracy_score(yte,base)*100:.1f}% F1={f1_score(yte,base,zero_division=0):.3f}")

# ---- leakage check on scaler
lr_pipe = Pipeline([("prep", prep()), ("clf", LogisticRegression(max_iter=1000))])
tmp = clone(prep()).fit(Xtr)
print(f"scaler learned mean of frequency (train only) = {tmp.named_transformers_['num'].mean_[NUM.index('frequency')]:.4f}  "
      f"full-cohort mean = {F.frequency.mean():.4f}")

# ---- tuning
gl = GridSearchCV(lr_pipe, {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(Xtr, ytr)
print(f"\nLogReg best params={gl.best_params_} cv F1={gl.best_score_:.4f}")
rf_pipe = Pipeline([("prep", prep()), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
gr = GridSearchCV(rf_pipe, {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]}, cv=5, scoring="f1").fit(Xtr, ytr)
print(f"RandomForest best params={gr.best_params_} cv F1={gr.best_score_:.4f}")

print("\nTest-set evaluation (lapsed = positive class):")
p_lr = metrics(gl.best_estimator_, "LogReg")
p_rf = metrics(gr.best_estimator_, "RandomForest")
f_lr, f_rf = f1_score(yte, p_lr), f1_score(yte, p_rf)
best_name, best = ("RandomForest", gr.best_estimator_) if f_rf >= f_lr else ("LogReg", gl.best_estimator_)
print(f"Selected model: {best_name}")

# ---- class balance experiment (training rows only)
rng_idx = np.random.RandomState(42)
tr = Xtr.copy(); tr["_y"] = ytr
maj, mino = tr[tr._y == 0], tr[tr._y == 1]
extra = mino.sample(n=len(maj) - len(mino), replace=True, random_state=42)
bal = pd.concat([tr, extra])
print(f"\nBalanced training set: {(bal._y==0).sum()} / {(bal._y==1).sum()}")
lr_best = clone(gl.best_estimator_).fit(bal[FEATS], bal._y)
p_bal = lr_best.predict(Xte)
for nm, p in [("LogReg unbalanced", p_lr), ("LogReg balanced", p_bal)]:
    print(f"{nm}: precision={precision_score(yte,p,zero_division=0):.3f} recall={recall_score(yte,p,zero_division=0):.3f} F1={f1_score(yte,p,zero_division=0):.3f}")

# ---- interpretation
names = prep().fit(Xtr).get_feature_names_out()
rf_imp = pd.Series(gr.best_estimator_.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
print("\nTop-5 random-forest importances:"); print(rf_imp.head(5).round(4).to_string())
coef = pd.Series(gl.best_estimator_.named_steps["clf"].coef_[0], index=names)
print("\nLogReg coefficients (selected):")
for f in ["num__recency", "num__frequency", "num__monetary", "num__cancel_rate", "num__avg_delivery"]:
    print(f"  {f}: {coef[f]:+.3f}")

# ---- persist + reload
joblib.dump(best, "outputs/lapse_model.joblib")
re = joblib.load("outputs/lapse_model.joblib")
print("\nreloaded model reproduces test predictions exactly:", bool((re.predict(Xte) == best.predict(Xte)).all()))

# ---- retention list (illustrative, post-hoc premium flag)
seg = pd.read_csv("outputs/customer_segments.csv").set_index("customer_id")
te = pd.DataFrame({"lapse_probability": re.predict_proba(Xte)[:, 1], "actually_lapsed": yte}, index=Xte.index)
te = te.join(seg.is_premium, how="left"); te["is_premium"] = te.is_premium.fillna(False).astype(bool)
ret = te[te.is_premium].sort_values("lapse_probability", ascending=False).head(10)
ret = ret.rename_axis("customer_id").reset_index()[["customer_id", "lapse_probability", "actually_lapsed"]]
ret["lapse_probability"] = ret.lapse_probability.round(4)
ret.to_csv("outputs/retention_list.csv", index=False)
print(f"premium customers in test set: {int(te.is_premium.sum())}; retention list rows={len(ret)}; truly lapsed={int(ret.actually_lapsed.sum())}")
