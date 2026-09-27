"""MSP-style confidence baselines for C2 -- all untrained, straight off C0's probability.

C0 predicts at its Youden threshold t, not at 0.5, so the textbook MSP (distance from 0.5)
does not measure distance from the decision C0 actually made. Three variants:

* ``standard``  -- max(p, 1-p). Hendrycks & Gimpel; ignores t.
* ``threshold`` -- p if p >= t else 1-p. The predicted class's probability under C0's own
                   decision rule. The two branches live on different ranges ((t,1] vs
                   (1-t,1]), so for t < 0.5 every predicted negative outranks a borderline
                   predicted positive -- which matches a low Youden t, where positive
                   predictions have low precision.
* ``logit``     -- MSP after shifting the logit so t maps to 0.5:
                   p' = sigmoid(logit(p) - logit(t)), score = max(p', 1-p'). Continuous,
                   minimal at t; equivalent in rank to |logit(p) - logit(t)|.

Every function returns a score where higher = more confident C0 is right.
"""
import numpy as np

EPS = 1e-7


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def msp_standard(p, t=None):
    """max(p, 1-p). `t` is accepted and ignored so all variants share one signature."""
    p = np.asarray(p, float)
    return np.maximum(p, 1.0 - p)


def msp_threshold(p, t):
    """Probability of the class C0 predicted at threshold t."""
    p = np.asarray(p, float)
    return np.where(p >= t, p, 1.0 - p)


def msp_logit(p, t):
    """Standard MSP of the prior-shifted probability sigmoid(logit(p) - logit(t))."""
    p_shift = 1.0 / (1.0 + np.exp(-(_logit(p) - _logit(t))))
    return np.maximum(p_shift, 1.0 - p_shift)


MSP_VARIANTS = {"standard": msp_standard, "threshold": msp_threshold, "logit": msp_logit}


def msp_score(p, t=None, variant="standard"):
    """Dispatch by name. `t` is required for every variant except ``standard``."""
    if variant not in MSP_VARIANTS:
        raise ValueError(f"unknown MSP variant {variant!r}; choose from {list(MSP_VARIANTS)}")
    if variant != "standard" and t is None:
        raise ValueError(f"MSP variant {variant!r} needs the C0 threshold t")
    return MSP_VARIANTS[variant](p, t)
