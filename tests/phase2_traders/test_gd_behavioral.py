import pytest
import numpy as np
from traders.legacy.gd import GD
from engine.market import Market
from traders.legacy.zic import ZIC

def test_gd_belief_formation_buyer():
    """
    Test that GD buyer forms correct beliefs q(b) from history.
    
    Scenario:
    - Bid 100 (Rejected) -> RB(100)=1
    - Bid 120 (Accepted) -> TB(120)=1
    - Ask 130 (Accepted) -> TA(130)=1
    
    Formula: q(b) = [TB(<=b) + A(<=b)] / [TB(<=b) + A(<=b) + RB(>b)]
    
    At b=100:
    - TB(<=100) = 0
    - A(<=100) = 0
    - RB(>100) = 0 (Wait, RB(100) is at 100. RB(>100) is 0)
    - q(100) = 0 / 0 -> 0.5 (fallback) or 0?
    
    Let's trace carefully:
    History:
    (100, True, False)  -> RB at 100
    (120, True, True)   -> TB at 120
    (130, False, True)  -> TA at 130
    
    Sorted prices: 100, 120, 130
    
    b=100:
    - TB(<=100) = 0
    - A(<=100) = 0
    - RB(>100) = 0
    - q(100) = 0 / 0 = 0.5 (fallback)
    
    b=120:
    - TB(<=120) = 1 (at 120)
    - A(<=120) = 0
    - RB(>120) = 0
    - q(120) = 1 / 1 = 1.0
    
    b=130:
    - TB(<=130) = 1
    - A(<=130) = 1 (TA at 130)
    - RB(>130) = 0
    - q(130) = 2 / 2 = 1.0
    """
    gd = GD(1, True, 1, [200], price_min=0, price_max=200)
    
    # Manually inject history
    gd.history = [
        (100, True, False), # Bid 100 Rejected
        (120, True, True),  # Bid 120 Accepted
        (130, False, True)  # Ask 130 Accepted
    ]
    
    prices, probs = gd._build_belief_curve(is_buyer=True)
    
    # Check specific points (using nearest price logic in test)
    # prices should contain 0, 100, 120, 130, 200
    
    # Find index of 120
    idx_120 = prices.index(120)
    assert probs[idx_120] == 1.0, f"q(120) should be 1.0, got {probs[idx_120]}"
    
    # Find index of 130
    idx_130 = prices.index(130)
    assert probs[idx_130] == 1.0, f"q(130) should be 1.0, got {probs[idx_130]}"

def test_gd_belief_formation_seller():
    """
    Test that GD seller forms correct beliefs p(a) from history.
    
    Scenario:
    - Ask 150 (Rejected) -> RA(150)=1
    - Ask 130 (Accepted) -> TA(130)=1
    - Bid 120 (Accepted) -> TB(120)=1
    
    Formula: p(a) = [TA(>=a) + B(>=a)] / [TA(>=a) + B(>=a) + RA(<=a)]
    
    Sorted prices: 120, 130, 150
    
    a=120:
    - TA(>=120) = 1 (at 130)
    - B(>=120) = 1 (TB at 120)
    - RA(<=120) = 0
    - p(120) = 2 / 2 = 1.0
    
    a=130:
    - TA(>=130) = 1 (at 130)
    - B(>=130) = 0
    - RA(<=130) = 0
    - p(130) = 1 / 1 = 1.0
    
    a=150:
    - TA(>=150) = 0
    - B(>=150) = 0
    - RA(<=150) = 1 (at 150)
    - p(150) = 0 / 1 = 0.0
    """
    gd = GD(1, False, 1, [100], price_min=0, price_max=200)
    
    gd.history = [
        (150, False, False), # Ask 150 Rejected
        (130, False, True),  # Ask 130 Accepted
        (120, True, True)    # Bid 120 Accepted
    ]
    
    prices, probs = gd._build_belief_curve(is_buyer=False)
    
    idx_120 = prices.index(120)
    assert probs[idx_120] == 1.0, f"p(120) should be 1.0, got {probs[idx_120]}"
    
    idx_130 = prices.index(130)
    assert probs[idx_130] == 1.0, f"p(130) should be 1.0, got {probs[idx_130]}"
    
    idx_150 = prices.index(150)
    assert probs[idx_150] == 0.0, f"p(150) should be 0.0, got {probs[idx_150]}"

def test_gd_quote_optimization():
    """
    Test that GD chooses price maximizing expected surplus.
    """
    gd = GD(1, True, 1, [200], price_min=0, price_max=200)
    
    # Mock belief curve:
    # Price 100: prob 1.0 -> Exp Surplus = 1.0 * (200 - 100) = 100
    # Price 150: prob 0.8 -> Exp Surplus = 0.8 * (200 - 150) = 40
    # Price 50:  prob 0.1 -> Exp Surplus = 0.1 * (200 - 50) = 15
    
    # We can't easily mock the internal method _build_belief_curve without patching,
    # so let's construct a history that produces this.
    # To get prob 1.0 at 100: Many accepted trades <= 100
    # To get prob low at 50: (Wait, for buyer, prob of acceptance increases with price)
    # q(b) is monotonic increasing.
    
    # History: 
    # 1. Ask 150 Accepted -> q(b)=1 for b>=150
    # 2. Bid 140 Rejected -> RB(>b) >= 1 for b < 140.
    #    At b=139: TB=0, A=0, RB=1. q(139) = 0/1 = 0.
    #    This forces the agent to bid at least 140+ to get non-zero prob.
    #    Actually, at 140, RB(>140)=0? No, RB is at 140. RB(>140) is 0.
    #    So at 140, q(140) is still 0/0 -> 0.5?
    #    Let's add Bid 149 Rejected.
    #    Then for b < 149, RB(>b) >= 1. q(b) = 0.
    
    gd.history = [
        (150, False, True),  # Ask 150 Accepted
        (140, True, False)   # Bid 140 Rejected
    ]
    
    # With Bid 140 Rejected:
    # b < 140: RB(>b) includes 140. So RB >= 1. q(b) = 0.
    # b = 140: RB(>140) = 0. q(140) = 0.5 (fallback).
    # b = 150: A(<=150) = 1. q(150) = 1.
    
    # Exp Surplus at 140: 0.5 * (200-140) = 30.
    # Exp Surplus at 150: 1.0 * (200-150) = 50.
    # Optimal should be 150.
    
    quote = gd._calculate_quote()
    
    # Note: Depending on interpolation, 145 might have prob 0.75 -> surplus 55 * 0.75 = 41.
    # So 150 should still win.
    
    assert quote == 150, f"Optimal bid should be 150, got {quote}"

def test_gd_market_interaction():
    """
    Test GD in a live market to ensure it trades.
    """
    # 1 GD Buyer vs 1 ZIC Seller
    # GD Val = 200, ZIC Cost = 100
    # ZIC will ask random [100, 200]
    # GD should eventually accept or bid
    
    buyer = GD(1, True, 1, [200], price_min=0, price_max=200)
    seller = ZIC(2, False, 1, [100], price_min=0, price_max=200)
    
    market = Market(
        num_buyers=1,
        num_sellers=1,
        num_times=50,
        price_min=0,
        price_max=200,
        buyers=[buyer],
        sellers=[seller]
    )
    
    # Run market
    trades = 0
    for _ in range(50):
        if market.run_time_step():
            trades += 1
            
    assert trades > 0, "GD failed to trade in 50 steps"
    assert buyer.num_trades > 0, "GD buyer did not record trade"