# ZIP Trigger Condition Verification

## Paper Figure 27 Pseudo-code (Cliff 1997)

### SELLERS:
```
if (last shout was ACCEPTED at price q)
  then:
    1. any seller si for which pi ≤ q should RAISE margin
    2. if (last shout was a BID)
       then: any ACTIVE seller si for which pi ≥ q should LOWER margin

else:  # last shout REJECTED
    1. if (last shout was an OFFER)
       then: any ACTIVE seller si for which pi ≥ q should LOWER margin
```

### BUYERS:
```
if (last shout was ACCEPTED at price q)
  then:
    1. any buyer bi for which pi ≥ q should RAISE margin
    2. if (last shout was an OFFER)
       then: any ACTIVE buyer bi for which pi ≤ q should LOWER margin

else:  # last shout REJECTED
    1. if (last shout was a BID)
       then: any ACTIVE buyer bi for which pi ≤ q should LOWER margin
```

## Our Implementation Analysis

### Current `_should_raise_margin` (lines 168-182):
```python
if not self.last_shout_accepted:
    return False  # ✓ Only raise if accepted

if self.is_buyer:
    return my_price >= trade_price  # ✓ CORRECT: pi ≥ q
else:
    return my_price <= trade_price  # ✓ CORRECT: pi ≤ q
```
**STATUS:** ✅ CORRECT

### Current `_should_lower_margin` (lines 184-229):
```python
if self.num_trades >= self.num_tokens:
    return False  # ✓ Only active agents

if self.last_shout_accepted:  # ACCEPTED case
    if self.is_buyer:
        if not self.last_shout_was_bid:  # last was OFFER
            return my_price <= last_price  # ✓ pi ≤ q
    else:
        if self.last_shout_was_bid:  # last was BID
            return my_price >= last_price  # ✓ pi ≥ q
else:  # REJECTED case
    if self.is_buyer:
        if self.player_id == high_bidder:
            return False  # Don't respond to own shout
        return my_price <= last_price  # ⚠️ Responds to ANY reject
    else:
        if self.player_id == low_asker:
            return False  # Don't respond to own shout
        return my_price >= last_price  # ⚠️ Responds to ANY reject
```

## Issues Found

### Issue 1: REJECTED case logic ⚠️
**Paper:** SELLERS only lower on rejected OFFER, BUYERS only lower on rejected BID
**Our Code:** Responds to ANY rejected shout (both bid and offer)

**Example Bug Scenario:**
1. Seller makes high offer (e.g., $10), gets rejected
2. Another seller with pi=9 sees reject
3. Paper: Should NOT lower (rejected shout was OFFER, same side)
4. Our code: LOWERS margin (sees ANY reject where pi≥q)
5. Result: Sellers compete with each other too aggressively

**Fix Needed:**
```python
else:  # REJECTED case
    if self.is_buyer:
        # Only respond to rejected BIDs (not OFFERs)
        if self.last_shout_was_bid:  # ← ADD THIS CHECK
            if self.player_id == high_bidder:
                return False
            return my_price <= last_price
    else:
        # Only respond to rejected OFFERs (not BIDs)
        if not self.last_shout_was_bid:  # ← ADD THIS CHECK
            if self.player_id == low_asker:
                return False
            return my_price >= last_price
```

### Issue 2: Self-shout prevention
**Current:** Checks `player_id` to avoid own shouts ✅
**Status:** GOOD - prevents divergence bug

## Summary

- ✅ Raise margin logic: CORRECT
- ✅ Lower margin (ACCEPTED case): CORRECT
- ⚠️ Lower margin (REJECTED case): BUG - responds to wrong shout types
- ✅ Active-only enforcement: CORRECT
- ✅ Self-shout prevention: GOOD ADDITION

## Recommendation
Fix REJECTED case to match paper Figure 27 exactly.
