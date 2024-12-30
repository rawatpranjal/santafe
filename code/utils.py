# utils.py

import numpy as np

def compute_equilibrium(buyer_vals, seller_costs):
    """
    Suppose we have a list of buyer valuations: e.g. [v1, v2, v3, ...] sorted descending
    and a list of seller costs: e.g. [c1, c2, c3, ...] sorted ascending.
    We find the largest Q where buyer_vals[Q-1] >= seller_costs[Q-1].
    eq price ~ (buyer_vals[Q-1] + seller_costs[Q-1]) / 2
    Also compute maximum theoretical surplus if Q>0.
    """
    nb = len(buyer_vals)
    ns = len(seller_costs)
    Qmax = min(nb, ns)

    eq_q = 0
    for q in range(1, Qmax+1):
        if buyer_vals[q-1] >= seller_costs[q-1]:
            eq_q = q
        else:
            break

    if eq_q==0:
        # no crossing
        eq_p = 0.5*(buyer_vals[-1] + seller_costs[0]) if buyer_vals and seller_costs else 0.5
        return 0, eq_p, 0.0

    eq_p = 0.5*(buyer_vals[eq_q-1] + seller_costs[eq_q-1])
    total = 0.0
    for i in range(eq_q):
        total += (buyer_vals[i] - seller_costs[i])
    return eq_q, eq_p, total
