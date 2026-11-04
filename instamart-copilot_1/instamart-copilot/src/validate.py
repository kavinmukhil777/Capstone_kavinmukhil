"""Part A validation: asserts the reference checkpoints from the brief."""
import pandas as pd
o = pd.read_csv("data/clean/orders.csv"); c = pd.read_csv("data/clean/customers.csv")
comp = o[o.status == "completed"]; canc = o[o.status == "cancelled"]
checks = [
    ("12,309 order rows", len(o) == 12309),
    ("one row per order_id", o.order_id.is_unique),
    ("11,110 completed", len(comp) == 11110),
    ("1,199 cancelled", len(canc) == 1199),
    ("no pending rows", (o.status == "pending").sum() == 0),
    ("0 missing order_amount", o.order_amount.isna().sum() == 0),
    ("0 missing delivery (completed)", comp.delivery_minutes.isna().sum() == 0),
    ("1,199 missing delivery (cancelled)", canc.delivery_minutes.isna().sum() == 1199),
    ("completed revenue 2,998,668.80", round(comp.order_amount.sum(), 2) == 2998668.80),
    ("six cities", c.city.nunique() == 6),
    ("city counts", c.city.value_counts().to_dict() == {"Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}),
    ("genders 610/560/30", c.gender.value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}),
    ("0 missing ages, all 18..60", c.age.isna().sum() == 0 and c.age.between(18, 60).all()),
]
for name, ok in checks:
    assert ok, f"FAILED: {name}"
    print(f"  ok  {name}")
print("ALL CHECKPOINTS PASSED")
