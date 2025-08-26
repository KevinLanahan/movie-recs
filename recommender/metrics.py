import numpy as np

def precision_recall_at_k(recs_by_user, test_items_by_user, k=10):

    precs, recs = [], []
    for u, recs_list in recs_by_user.items():
        truth = test_items_by_user.get(u, set())
        if not truth: continue
        topk = recs_list[:k]
        hit = len([m for m in topk if m in truth])
        precs.append(hit / k)
        recs.append(hit / max(1, len(truth)))
    return float(np.mean(precs) if precs else 0.0), float(np.mean(recs) if recs else 0.0)
