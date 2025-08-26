def blend(cf_list, content_list, alpha=0.7, top_k=10):
    from collections import defaultdict
    s = defaultdict(float)
    for m,sc in cf_list: s[m] += alpha*sc
    for m,sc in content_list: s[m] += (1-alpha)*sc
    items = sorted(s.items(), key=lambda x: -x[1])[:top_k]
    return items
