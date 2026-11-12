# check_traj_start.py
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from c2_train_trajectory import load_data
from c2_msp import msp_score

for d in ["atelectasis", "cardiomegaly", "consolidation", "edema", "effusion"]:
    img, P, thr = load_data(d)
    traj = P[:, 2, :]                         # high level, 30 recorded values
    p, y = img["orig_prob"].values, img["correct"].values
    start, end = traj[:, 0], traj[:, -1]
    total = np.abs(end - p)
    done = np.abs(start - p) / np.where(total > 1e-6, total, np.nan)
    print(f"{d:14s} moved before 1st record: median {np.nanmedian(done):.0%}  "
          f"spearman(start,p)={spearmanr(start, p).correlation:.3f}  "
          f"adjMSP(p)={roc_auc_score(y, msp_score(p, thr, 'threshold')):.4f}  "
          f"adjMSP(start)={roc_auc_score(y, msp_score(start, thr, 'threshold')):.4f}")