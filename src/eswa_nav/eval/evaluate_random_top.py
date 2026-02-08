import json, glob
from evaluate_scenario import load_model, evaluate_scenario

MODEL="checkpoints_rl_only/ppo_static_shaping_model.zip"
VEC="checkpoints_rl_only/ppo_static_shaping_vecnorm.pkl"
REPEATS=20

model, venv = load_model(MODEL, VEC)

rows=[]
for fp in sorted(glob.glob("scenarios/random_top/top_*.json")):
    sc = json.load(open(fp,"r"))
    episodes, agg = evaluate_scenario(
        model, venv, sc,
        repeats=REPEATS,
        seed_base=int(sc["seed"])*10
    )
    rows.append({"scenario_id": sc["scenario_id"], **agg})

out="scenarios/random_top/eval_summary_R20.json"
json.dump(rows, open(out,"w"), indent=2)
print("Wrote", out)

