import numpy as np

def _val_norm_mask(y_true, g):
    y = np.asarray(y_true, dtype=int)
    gg = np.asarray(g, dtype=int)
    return (y == 0) & (gg == 0)

def far_at_threshold(scores_norm, thr):
    s = np.asarray(scores_norm, float)
    return float((s > thr).mean())

def select_threshold_by_far(y_true, g, scores, far_target=0.05, mode='nearest'):
    """
    Calibrate threshold to hit FAR ~= far_target on validation NORMALS.
    mode: 'nearest' | 'ceil' (FAR<=target) | 'floor' (FAR>=target)
    """
    y = np.asarray(y_true, int); gg = np.asarray(g, int); s = np.asarray(scores, float)
    m = _val_norm_mask(y, gg)
    s_norm = np.sort(s[m])
    if s_norm.size == 0:
        raise ValueError("No validation normals available for FAR calibration.")

    # Raw quantile proposal
    q = 1.0 - float(np.clip(far_target, 0.0, 1.0))
    k = int(np.floor(q * (s_norm.size - 1)))
    k = np.clip(k, 0, s_norm.size - 1)
    thr = s_norm[k]

    # Evaluate neighbors for 'nearest'
    far_thr = far_at_threshold(s_norm, thr)
    k_up = min(k + 1, s_norm.size - 1)
    thr_up = s_norm[k_up]; far_up = far_at_threshold(s_norm, thr_up)

    if mode == 'ceil':
        # ensure FAR <= target
        if far_thr > far_target:
            thr = np.nextafter(thr, np.inf)
    elif mode == 'floor':
        if far_thr < far_target:
            thr = np.nextafter(thr, -np.inf)
    else:  # nearest
        if abs(far_up - far_target) < abs(far_thr - far_target):
            thr = thr_up

    achieved = far_at_threshold(s_norm, thr)
    return dict(thr=float(thr), achieved_far=float(achieved), n_norm=int(s_norm.size))

def select_threshold_f1_with_far_cap(y_true, g, scores, far_cap=None):
    """
    For baselines: maximize overall F1 on validation (optionally subject to FAR<=far_cap).
    """
    from sklearn.metrics import f1_score
    y = np.asarray(y_true, int); s = np.asarray(scores, float)
    uniq = np.unique(s)
    candidates = np.concatenate(([-np.inf], uniq, [np.inf]))

    # Optionally enforce FAR cap using validation normals only
    yy = np.asarray(y_true, int); gg = np.asarray(g, int)
    m_norm = _val_norm_mask(yy, gg)
    s_norm = s[m_norm]

    best = (-1.0, 0.0, 1.0)  # (f1, thr, far_val)
    eps = 1e-12
    for t in candidates:
        yhat = (s > t).astype(int)
        f1 = f1_score(y, yhat, zero_division=0)
        far_val = float((s_norm > t).mean()) if s_norm.size else 0.0
        if far_cap is not None and (far_val - far_cap) > 1e-12:
            continue
        if f1 > best[0] + eps:
            best = (f1, float(t), far_val)
    # If none met cap, fall back to pure F1
    if best[0] < 0.0:
        for t in candidates:
            yhat = (s > t).astype(int)
            f1 = f1_score(y, yhat, zero_division=0)
            far_val = float((s_norm > t).mean()) if s_norm.size else 0.0
            if f1 > best[0]:
                best = (f1, float(t), far_val)
    return dict(thr=best[1], val_F1=float(best[0]), achieved_far=float(best[2]))

def select_threshold_sr_targetDR_with_far_cap(y_true, g, scores, far_cap=None, target_type=1):
    """
    For SR-DSVDD: among thresholds satisfying FAR<=far_cap (if set), choose the one
    maximizing detection rate (recall) on the targeted anomaly type (g==target_type).
    Ties broken by overall F1.
    """
    from sklearn.metrics import f1_score
    y = np.asarray(y_true, int); gg = np.asarray(g, int); s = np.asarray(scores, float)
    uniq = np.unique(s)
    candidates = np.concatenate(([-np.inf], uniq, [np.inf]))

    # FAR constraint computed on validation normals
    m_norm = _val_norm_mask(y, gg)
    s_norm = s[m_norm]

    # Mask for targeted anomalies
    m_t = (gg == target_type) & (y == 1)

    best = (-1.0, -1.0, 0.0)  # (rec_t, f1_overall, thr)
    for t in candidates:
        yhat = (s > t).astype(int)
        if far_cap is not None and s_norm.size:
            far_val = float((s_norm > t).mean())
            if far_val - far_cap > 1e-12:
                continue
        # targeted detection rate
        tp_t = float((yhat[m_t] == 1).sum()); tot_t = float(m_t.sum())
        rec_t = tp_t / max(tot_t, 1.0)
        f1 = f1_score(y, yhat, zero_division=0)
        key = (rec_t, f1, float(t))
        if key > best:
            best = key
    # If nothing met FAR cap, fall back to best targeted DR ignoring cap
    if best[0] < 0.0:
        for t in candidates:
            yhat = (s > t).astype(int)
            tp_t = float((yhat[m_t] == 1).sum()); tot_t = float(m_t.sum())
            rec_t = tp_t / max(tot_t, 1.0)
            f1 = f1_score(y, yhat, zero_division=0)
            key = (rec_t, f1, float(t))
            if key > best:
                best = key

    thr = best[2]
    achieved_far = float((s_norm > thr).mean()) if s_norm.size else 0.0
    yhat = (s > thr).astype(int)
    from sklearn.metrics import f1_score
    f1_val = f1_score(y, yhat, zero_division=0)
    return dict(thr=float(thr), val_DR_target=float(best[0]),
                val_F1=float(f1_val), achieved_far=achieved_far)
