"""Pairs of cardiomegaly calls with the same C0 p, one right and one wrong, in different groups.

Builds case_card_combined.ipynb's joined table J (its first three code cells: setup, both arms,
join), then lists the pairs. Run from final_experiment/ on a GPU node (the saliency arm computes
Grad-CAMs):  python -i code/find_matched_pairs.py
"""
import os, json
import pandas as pd

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # final_experiment/, from anywhere
os.environ.setdefault("XAI_DISEASE", "cardiomegaly")
PAIR_TOL = 0.005                     # max |p_right - p_wrong|

cells = [c for c in json.load(open("case_card_combined.ipynb"))["cells"] if c["cell_type"] == "code"]
ns = {"__name__": "__main__", "display": print}
for c in cells[:3]:
    exec(compile("".join(c["source"]), "<case_card_combined>", "exec"), ns)
J, LAST = ns["J"], ns["LETTERS"][ns["N_CLUSTERS"] - 1]

# Both held out in both arms, same side of C0's call. Right one: group A in both arms. Wrong one:
# never A, and the last (most errors) group in at least one arm.
H    = J[J["held_out_attr"] & J["held_out_img"]].reset_index()
cols = ["path", "prob", "outcome", "group_attr", "group_img", "ens_rank", "msp_rank"]
PAIRS = {}
for side, name in ns["SIDE_NAME"].items():
    h = H[H["side"] == side]
    R = h[(h["correct"] == 1) & (h["group_attr"] == "A") & (h["group_img"] == "A")][cols]
    W = h[(h["correct"] == 0) & (h["group_attr"] != "A") & (h["group_img"] != "A")
          & ((h["group_attr"] == LAST) | (h["group_img"] == LAST))][cols]
    m = R.merge(W, how="cross", suffixes=("_R", "_W"))
    m["dp"] = (m["prob_R"] - m["prob_W"]).abs()
    PAIRS[side] = m[m["dp"] < PAIR_TOL].sort_values("dp").reset_index(drop=True)
    print(f"\n{name}: {len(R)} right in A/A, {len(W)} wrong in {LAST}, "
          f"{len(PAIRS[side])} pairs with |dp| < {PAIR_TOL}")
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(PAIRS[side].head(10).round(3).to_string())
