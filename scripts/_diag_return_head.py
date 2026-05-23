"""One-shot diagnostic: Check 1 (return label coverage) + Check 2 (log_var suppression)."""

import sqlite3, torch
from pathlib import Path

DB = Path(".tirra_pipeline/pipeline.db")
MODEL = Path(".tirra_pipeline/gnn_model.pt")

con = sqlite3.connect(DB)
cur = con.cursor()

print("=" * 60)
print("CHECK 1: Instrument return label coverage")
print("=" * 60)

cur.execute("SELECT COUNT(*) FROM entities WHERE entity_type = 'instrument'")
print(f"Total instrument entities: {cur.fetchone()[0]}")

cur.execute("""
    SELECT o.observation_type, COUNT(DISTINCT o.entity_id) as n_ent, COUNT(*) as n_obs
    FROM entity_observations o
    JOIN entities e ON o.entity_id = e.entity_id
    WHERE e.entity_type = 'instrument'
    GROUP BY o.observation_type
    ORDER BY n_obs DESC
    LIMIT 20
""")
rows = cur.fetchall()
print("\nObs types for instrument entities:")
for obs_type, n_ent, n_obs in rows:
    print(f"  {obs_type:<30} {n_ent:>4} entities  {n_obs:>8} obs")

cur.execute("""
    SELECT COUNT(DISTINCT o.entity_id), COUNT(*), MIN(o.observed_at), MAX(o.observed_at)
    FROM entity_observations o
    JOIN entities e ON o.entity_id = e.entity_id
    WHERE e.entity_type = 'instrument'
      AND o.observation_type = 'log_return'
""")
row = cur.fetchone()
if row[1]:
    print(
        f"\nlog_return: {row[1]} obs across {row[0]} instruments, range {row[2]:.0f} to {row[3]:.0f}"
    )
else:
    print("\nNO log_return obs found for instruments")

cur.execute("""
    SELECT o.observation_type, COUNT(*) as n
    FROM entity_observations o
    JOIN entities e ON o.entity_id = e.entity_id
    WHERE e.entity_type = 'instrument'
      AND o.observation_type IN ('price_close','ohlcv','price','close','return','log_return','pct_return')
    GROUP BY o.observation_type
""")
print("Price/return obs types:", cur.fetchall() or "NONE found")
con.close()

print("\n" + "=" * 60)
print("CHECK 2: log_var values in saved model")
print("=" * 60)

ckpt = torch.load(MODEL, map_location="cpu", weights_only=False)
state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
log_vars = {k: v for k, v in state.items() if "log_var" in k}
if log_vars:
    print("log_var parameters:")
    for k, v in log_vars.items():
        val = float(v.item())
        weight = float(torch.exp(-v).item())
        print(f"  {k:<40} s={val:+.4f}  exp(-s)={weight:.6f}")
else:
    print("No log_var params — auto-tune was NOT active in this checkpoint")
    relevant = [k for k in state.keys() if "return" in k.lower()]
    print("return-related keys:", relevant[:10])
