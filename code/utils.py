# utils.py
import random
import logging
import math
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
try:
    # Try using a non-interactive backend suitable for servers or scripts
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker # For formatting ticks
    plt_available = True
except ImportError:
    plt = None # Plotting will be disabled
    mticker = None
    plt_available = False
    logging.error("Matplotlib not found or backend error. Plotting functions will be disabled.")

from tabulate import tabulate
import os
import ast # Use ast.literal_eval instead of eval for safety


# --- Plotting Style ---
if plt_available:
    plt.style.use('ggplot')
    matplotlib.rcParams.update({'font.size': 8})


# --- Equilibrium Calculation ---
def compute_equilibrium(buyer_vals, seller_costs):
    """
    Compute theoretical equilibrium quantity, price range, and max surplus.
    Assumes values/costs are sorted appropriately (buyers descending, sellers ascending).
    Handles integer inputs. Returns representative midpoint price.
    """
    logger = logging.getLogger('utils.equilibrium')
    # Ensure input lists are not empty and contain numbers
    # Convert numpy arrays if necessary
    if isinstance(buyer_vals, np.ndarray): buyer_vals = buyer_vals.tolist()
    if isinstance(seller_costs, np.ndarray): seller_costs = seller_costs.tolist()

    if not buyer_vals or not all(isinstance(v, (int, float, np.number)) for v in buyer_vals):
        logger.warning("Cannot compute equilibrium with invalid buyer values.")
        return 0, 0.0, 0.0
    if not seller_costs or not all(isinstance(c, (int, float, np.number)) for c in seller_costs):
        logger.warning("Cannot compute equilibrium with invalid seller costs.")
        return 0, 0.0, 0.0

    # Sort just in case they aren't pre-sorted
    sorted_buyers = sorted([float(v) for v in buyer_vals], reverse=True)
    sorted_sellers = sorted([float(c) for c in seller_costs])

    nb = len(sorted_buyers)
    ns = len(sorted_sellers)
    max_possible_q = min(nb, ns)

    eq_q = 0
    total_surplus = 0.0

    # Find equilibrium quantity and calculate surplus along the way
    for q in range(1, max_possible_q + 1):
        buyer_val_at_q = sorted_buyers[q - 1]
        seller_cost_at_q = sorted_sellers[q - 1]
        if buyer_val_at_q >= seller_cost_at_q:
            eq_q = q
            total_surplus += (buyer_val_at_q - seller_cost_at_q)
        else:
            # This unit (and subsequent units) won't trade
            break

    # Determine equilibrium price (midpoint of the competitive range)
    if eq_q == 0:
        # No trades possible. Price is indeterminate. Define range based on best potential.
        price_low = sorted_sellers[0] if ns > 0 else 0
        price_high = sorted_buyers[0] if nb > 0 else 0
        # Ensure low <= high
        eq_p_range = (min(price_low, price_high), max(price_low, price_high))
    else:
        # Price is bounded by the value/cost of the marginal traders involved in the eq_q'th trade
        # and potentially the value/cost of the first excluded traders (q+1)
        price_low_bound = sorted_sellers[eq_q - 1] # Seller q cost
        price_high_bound = sorted_buyers[eq_q - 1] # Buyer q value

        # Consider the next potential traders (q+1)
        next_buyer_val = sorted_buyers[eq_q] if eq_q < nb else -np.inf # Effectively no lower bound from next buyer
        next_seller_cost = sorted_sellers[eq_q] if eq_q < ns else np.inf # Effectively no upper bound from next seller

        # Price must be <= Buyer q's value AND <= next Seller's cost (if they exist)
        price_upper = min(price_high_bound, next_seller_cost)
        # Price must be >= Seller q's cost AND >= next Buyer's value (if they exist)
        price_lower = max(price_low_bound, next_buyer_val)

        # Ensure correct range ordering
        eq_p_range = (min(price_lower, price_upper), max(price_lower, price_upper))

    # Calculate midpoint, ensure it's float
    eq_p_mid = float(0.5 * (eq_p_range[0] + eq_p_range[1]))

    # Check for potential source of Efficiency > 1
    if total_surplus < 1e-9 and eq_q > 0:
        logger.warning(f"Calculated eq_q={eq_q} but eq_surplus is near zero ({total_surplus:.4f}). Check S/D values.")

    logger.debug(f"Equilibrium calculated: Q={eq_q}, P_range=({eq_p_range[0]:.2f}, {eq_p_range[1]:.2f}), P_mid={eq_p_mid:.2f}, Max Surplus={total_surplus:.2f}")
    return eq_q, eq_p_mid, total_surplus


# --- SFI Value Generation Functions ---
def generate_sfi_components(gametype, min_price, max_price, value_rng):
    """
    Generates the common random components (R, A) for SFI value calculation,
    using the provided random number generator instance.
    Based on SFI Rust/Palmer/Miller 1992 paper description, using R=max(0, 3^k-1).
    Returns: A dictionary {'R': [R1, R2, R3, R4], 'A': A_component}
    """
    logger = logging.getLogger('utils.sfi_values')
    logger.debug(f"Generating SFI components for gametype={gametype} using provided RNG")
    k = [0, 0, 0, 0] # Default k-values (digits)
    R = [0, 0, 0, 0] # Default R-values (bounds)

    try:
        gt_int = int(round(gametype))
        if not (0 <= gt_int <= 9999): # Check if conceptually 4 digits
            raise ValueError("Gametype should represent a 4-digit number (0-9999).")

        gt_str = f"{gt_int:04d}" # Format as 4 digits
        k = [int(digit) for digit in gt_str] # Extract digits directly

        # Use footnote 14 formula: Ri = 3^k(i) - 1, ensuring non-negative
        R_raw = [(3**ki - 1) for ki in k]
        R = [max(0, r_val) for r_val in R_raw] # Ensure R >= 0 for U[0, R]

        logger.debug(f"Gametype {gametype} -> Digits k={k} -> R={R} (Using R=max(0, 3^k-1))")

    except Exception as e:
        logger.error(f"Error processing gametype {gametype}: {e}. Defaulting to R=[0,0,0,0].", exc_info=True)
        R = [0, 0, 0, 0] # Fallback

    R1, _, _, _ = R
    # A ~ U[0, R1]. randint is inclusive [a, b].
    A = value_rng.randint(0, R1) if R1 >= 0 else 0 # Handle R1=0 case (or potentially <0 if max wasn't used)
    logger.debug(f"Generated SFI components: R={R}, A={A}")
    return {'R': R, 'A': A}

def calculate_sfi_values_for_participant(
    participant_name, role_l, N, sfi_components, min_price, max_price, value_rng):
    """
    Calculates N private values using SFI formula based on Rust/Palmer/Miller 1992 Eq 3.1.
    Buyer Tjk = A + B + Ck + Djk
    Seller Tjk = A + Ck + Djk
    Where A~U[0,R1], B~U[0,R2], Ck~U[0,R3], Djk~U[0,R4]
    Uses R values from generate_sfi_components.
    """
    logger = logging.getLogger('utils.sfi_values')
    R = sfi_components['R'] # Use R = [R1, R2, R3, R4] calculated via 3^k-1
    A = sfi_components['A']
    _, R2, R3, R4 = R # R1 used for A, R2 for B, R3 for Ck, R4 for Djk
    values = []

    # B component common for all tokens of a specific BUYER
    B_comp = value_rng.randint(0, R2) if R2 >= 0 else 0 # randint requires upper bound >= lower bound

    for k_token in range(1, N + 1): # Tokens indexed 1 to N
        Ck = value_rng.randint(0, R3) if R3 >= 0 else 0 # C component per token
        Djk = value_rng.randint(0, R4) if R4 >= 0 else 0 # D component per trader per token

        if role_l == 1: # Buyer
            T_jkl = A + B_comp + Ck + Djk
        elif role_l == 2: # Seller
            T_jkl = A + Ck + Djk # Seller doesn't get B component
        else:
            logger.error(f"Unknown role_l: {role_l}"); T_jkl = 0

        # Clamp value/cost to be within market bounds and ensure integer
        T_jkl_clamped = int(round(max(min_price, min(max_price, T_jkl))))

        if T_jkl != T_jkl_clamped and logger.isEnabledFor(logging.DEBUG):
             logger.debug(f"Value {T_jkl:.2f} clamped/rounded to {T_jkl_clamped} for P:{participant_name}, role {role_l}, token {k_token}")
        values.append(T_jkl_clamped)

    logger.debug(f"Calculated values/costs for P:{participant_name} (role {role_l}): {values}")
    return values


# --- Analysis Functions ---
def safe_literal_eval(val):
    """Safely evaluate string representations of lists/dicts."""
    if isinstance(val, (list, dict)): return val # Already evaluated
    if isinstance(val, str):
        try: return ast.literal_eval(val)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError): return None # Return None on error
    return None # Return None for other non-string types

def analyze_individual_performance(round_stats):
    """
    Aggregates bot performance across rounds and returns a formatted table string.
    Profit reported is the total profit accumulated by the end of each round.
    """
    logger = logging.getLogger('analysis.individual')
    all_bots_profit = defaultdict(list)
    if not round_stats: return "No round stats available for individual performance analysis."
    if not isinstance(round_stats, list) or not all(isinstance(rs, dict) for rs in round_stats):
         logger.error("Invalid format for round_stats - expected list of dictionaries.")
         return "Invalid round_stats format for individual performance analysis."

    for rstat in round_stats:
        # Safely evaluate bot_details if it's a string
        bot_details = safe_literal_eval(rstat.get("bot_details"))

        if not isinstance(bot_details, list):
            logger.debug(f"No valid bot details found for round {rstat.get('round', '?')}: {bot_details}")
            continue

        for b in bot_details:
            if not isinstance(b, dict): logger.warning(f"Invalid item in bot_details (expected dict): {b}"); continue
            role, strategy, name = b.get("role"), b.get("strategy"), b.get("name")
            profit_val = b.get("profit", 0.0)
            if role is None or strategy is None or name is None: logger.warning(f"Missing key fields in bot detail: {b}"); continue

            key = (role, strategy, name) # Unique key per bot instance
            try:
                profit = float(profit_val)
                all_bots_profit[key].append(profit)
            except (ValueError, TypeError):
                logger.warning(f"Invalid profit value '{profit_val}' for {key} in round {rstat.get('round', '?')}")

    if not all_bots_profit: return "No valid bot details found for individual performance analysis."

    table_rows = []
    # Sort by role, then strategy, then name for consistent output
    for (role, strategy, name), profit_list in sorted(all_bots_profit.items(), key=lambda item: item[0]):
        arr = np.array(profit_list)
        if len(arr) == 0: continue # Should not happen with defaultdict, but safety check
        avg_p, std_p, min_p, med_p, max_p = np.mean(arr), np.std(arr), np.min(arr), np.median(arr), np.max(arr)
        # Format numbers to 2 decimal places for readability
        table_rows.append([role, strategy, name, f"{avg_p:.2f}", f"{std_p:.2f}", f"{min_p:.2f}", f"{med_p:.2f}", f"{max_p:.2f}"])

    headers = ["Role", "Strategy", "BotName", "MeanProfit", "StdProfit", "MinProfit", "MedianProfit", "MaxProfit"]
    title = "\n=== INDIVIDUAL BOT PERFORMANCE (ACROSS ALL ROUNDS) ==="
    if not table_rows: return f"{title}\nNo data to display."
    # Use tabulate for nice formatting
    table = tabulate(table_rows, headers=headers, tablefmt="pretty", floatfmt=".2f")
    return f"{title}\n{table}"


def analyze_market_performance(round_stats):
    """Aggregates market performance across rounds and returns a formatted table string."""
    logger = logging.getLogger('analysis.market')
    if not round_stats or not isinstance(round_stats, list):
        logger.warning("No round stats available for market performance analysis.")
        return "No round stats available for market performance analysis."

    effs, price_diffs, quant_diffs = [], [], []
    buyer_surplus_frac_list, seller_surplus_frac_list = [], []
    eff_calc_issues = 0

    for rstat in round_stats:
        if not isinstance(rstat, dict): continue
        # Collect efficiency, price diff, quant diff
        try:
             eff = float(rstat.get('market_efficiency', 0.0))
             # Check for potentially erroneous efficiency values
             if eff < -0.01 or eff > 1.01: # Allow small floating point errors around 0 and 1
                  eff_calc_issues += 1
                  logger.debug(f"Round {rstat.get('round','?')}: Suspicious efficiency value: {eff:.4f}. Profit={rstat.get('actual_total_profit','N/A')}, EqSurplus={rstat.get('eq_surplus','N/A')}")
                  # Clamp efficiency for averaging, but log the issue
                  eff = np.clip(eff, 0.0, 1.0)
             effs.append(eff)
        except (ValueError, TypeError): pass
        adp = rstat.get('abs_diff_price');
        if adp is not None:
             try: price_diffs.append(float(adp))
             except (ValueError, TypeError): pass
        adq = rstat.get('abs_diff_quantity');
        if adq is not None:
             try: quant_diffs.append(float(adq))
             except (ValueError, TypeError): pass

        # Calculate buyer/seller surplus fraction
        bot_details = safe_literal_eval(rstat.get("bot_details"))
        if isinstance(bot_details, list):
            buyer_pft = sum(float(b.get("profit", 0.0)) for b in bot_details if isinstance(b,dict) and b.get("role") == "buyer")
            seller_pft = sum(float(s.get("profit", 0.0)) for s in bot_details if isinstance(s,dict) and s.get("role") == "seller")
            calc_total_pft = buyer_pft + seller_pft
            if calc_total_pft > 1e-9:
                buyer_surplus_frac_list.append(buyer_pft / calc_total_pft)
                seller_surplus_frac_list.append(seller_pft / calc_total_pft)
            else: # Avoid division by zero, assume equal split if no profit
                buyer_surplus_frac_list.append(0.5)
                seller_surplus_frac_list.append(0.5)
        else: # Append defaults if bot_details was invalid
            buyer_surplus_frac_list.append(0.5)
            seller_surplus_frac_list.append(0.5)

    if eff_calc_issues > 0:
         logger.warning(f"Encountered {eff_calc_issues} rounds with efficiency outside [0, 1]. Check equilibrium surplus calculation.")

    # Calculate aggregate statistics
    avg_eff = np.mean(effs) if effs else 0.0; std_eff = np.std(effs) if effs else 0.0
    avg_price_diff = np.mean(price_diffs) if price_diffs else 0.0
    avg_quant_diff = np.mean(quant_diffs) if quant_diffs else 0.0
    avg_buyer_surplus = np.mean(buyer_surplus_frac_list) if buyer_surplus_frac_list else 0.5
    avg_seller_surplus = np.mean(seller_surplus_frac_list) if seller_surplus_frac_list else 0.5

    table_rows = [[
        f"{avg_eff:.4f}", f"{std_eff:.4f}",
        f"{avg_buyer_surplus*100:.1f}%", f"{avg_seller_surplus*100:.1f}%",
        f"{avg_price_diff:.2f}", f"{avg_quant_diff:.2f}"
    ]]
    headers = ["MarketEff(Mean)", "MarketEff(Std)", "BuyerSurplus%", "SellerSurplus%", "AvgPriceDiff", "AvgQuantDiff"]
    title = "\n=== MARKET PERFORMANCE (AGGREGATE ACROSS ROUNDS) ==="
    table = tabulate(table_rows, headers=headers, tablefmt="pretty")
    if eff_calc_issues > 0: table += f"\nNote: Efficiency values were clamped to [0, 1] for {eff_calc_issues} rounds."
    return f"{title}\n{table}"


# --- Plotting Functions ---
def plot_per_round(round_stats, exp_path, dfLogs=None, generate_plots=True):
    """Generates a 3-panel plot for each round, if generate_plots is True."""
    logger = logging.getLogger('plotting.round')
    if not plt_available: logger.warning("Plotting disabled."); return
    if not generate_plots: logger.info("Skipping generation of per-round plots as per config."); return
    if dfLogs is None or dfLogs.empty: logger.warning("No step logs available for per-round plotting."); return
    if not round_stats: logger.warning("No round stats available for per-round plotting."); return

    num_buyers, num_sellers = 5, 5 # Default fallback
    try:
        first_round_details = safe_literal_eval(round_stats[0].get("bot_details"))
        if isinstance(first_round_details, list):
             n_b = sum(1 for d in first_round_details if isinstance(d, dict) and d.get('role') == 'buyer')
             n_s = sum(1 for d in first_round_details if isinstance(d, dict) and d.get('role') == 'seller')
             if n_b > 0: num_buyers = n_b
             if n_s > 0: num_sellers = n_s
             if num_buyers == 0 or num_sellers == 0: raise ValueError("Zero buyers or sellers inferred.")
             logger.debug(f"Inferred {num_buyers} buyers and {num_sellers} sellers for plotting.")
        else: raise ValueError("Could not parse bot details")
    except Exception as e: logger.warning(f"Could not infer num_buyers/sellers: {e}. Using fallback {num_buyers}/{num_sellers}.")

    logger.info(f"Attempting to generate per-round plots for {len(round_stats)} rounds...")
    plot_count = 0
    # Convert round column once before loop if possible
    try: dfLogs['round'] = pd.to_numeric(dfLogs['round'])
    except Exception as e: logger.error(f"Could not convert 'round' column to numeric in dfLogs: {e}"); return

    for rstat in round_stats:
        rnum = rstat.get("round", -1)
        if rnum == -1: continue
        fig = None # Initialize fig
        try:
            if 'round' not in dfLogs.columns: logger.warning(f"Missing 'round' column in step logs. Skipping plot {rnum}."); continue

            df_round = dfLogs[dfLogs["round"] == rnum].copy()

            if df_round.empty: logger.debug(f"No step logs found for round {rnum}. Skipping plot."); continue

            sort_cols = ['period', 'step']
            if not all(col in df_round.columns for col in sort_cols): logger.warning(f"Missing sort columns {sort_cols} in step logs. Skipping plot {rnum}."); continue
            # Convert period/step to numeric for sorting if they aren't already
            try:
                df_round['period'] = pd.to_numeric(df_round['period'])
                df_round['step'] = pd.to_numeric(df_round['step'])
            except Exception as e: logger.error(f"Could not convert period/step to numeric: {e}"); continue
            df_round.sort_values(by=sort_cols, inplace=True)

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            num_periods = rstat.get("num_periods", "?") # Get num periods for title
            fig.suptitle(f"Round {rnum} Analysis ({num_periods} Periods)", fontsize=12)

            # --- (1) Price Evolution ---
            ax_price = axes[0]
            step_indices = np.arange(len(df_round)) # Use simple index for x-axis
            eq_p = rstat.get("eq_p")
            cbid_series = pd.to_numeric(df_round["current_bid_price"], errors='coerce')
            cask_series = pd.to_numeric(df_round["current_ask_price"], errors='coerce')
            trade_vals = pd.to_numeric(df_round["trade_price"], errors='coerce')
            trade_mask = ~pd.isna(trade_vals) # Use pd.isna for Series
            ax_price.plot(step_indices, cbid_series, label="Current Bid", color='blue', linestyle='-', marker='.', markersize=2, alpha=0.6)
            ax_price.plot(step_indices, cask_series, label="Current Ask", color='red', linestyle='-', marker='.', markersize=2, alpha=0.6)
            if trade_mask.any(): ax_price.scatter(step_indices[trade_mask], trade_vals[trade_mask], label="Trade Price", color='green', marker='o', s=25, zorder=5)
            if eq_p is not None: ax_price.axhline(eq_p, color='grey', linestyle=':', label=f"Eq.Price ({eq_p:.0f})")
            ax_price.set_title(f"Market Quotes & Trades"); ax_price.set_xlabel("Step Index (Across Periods)"); ax_price.set_ylabel("Price"); ax_price.legend(fontsize=7); ax_price.grid(True, which='both', linestyle='--', linewidth=0.5)

            # --- (2) Supply vs Demand ---
            ax_sd = axes[1]
            buyer_vals = safe_literal_eval(rstat.get("buyer_vals"))
            seller_vals = safe_literal_eval(rstat.get("seller_vals"))
            eq_q = rstat.get("eq_q")
            plot_sd = False
            if isinstance(buyer_vals, list) and buyer_vals and isinstance(seller_vals, list) and seller_vals:
                try:
                    buyer_vals_num = pd.to_numeric(buyer_vals, errors='coerce').dropna().sort_values(ascending=False).values
                    seller_vals_num = pd.to_numeric(seller_vals, errors='coerce').dropna().sort_values().values
                    if len(buyer_vals_num) > 0 and len(seller_vals_num) > 0: plot_sd = True
                    else: logger.warning(f"R{rnum}: Numeric conversion yielded empty S/D arrays.")
                except Exception as e_num: logger.warning(f"R{rnum}: Could not convert S/D values: {e_num}.")
            if plot_sd:
                ax_sd.step(np.arange(1, len(buyer_vals_num) + 1), buyer_vals_num, where='pre', color='blue', label="Demand")
                ax_sd.step(np.arange(1, len(seller_vals_num) + 1), seller_vals_num, where='pre', color='red', label="Supply")
                if eq_p is not None: ax_sd.axhline(eq_p, color='grey', linestyle=':', label="Eq.Price")
                if eq_q is not None: ax_sd.axvline(eq_q + 0.5, color='grey', linestyle=':', label="Eq.Qty")
                trades_only = df_round[df_round["trade_executed"] == 1]; trade_prices_plot = pd.to_numeric(trades_only["trade_price"], errors='coerce').dropna().values; trade_quantities_plot = np.arange(1, len(trade_prices_plot) + 1)
                if len(trade_prices_plot) > 0: ax_sd.scatter(trade_quantities_plot, trade_prices_plot, color='green', marker='x', s=40, label='Actual Trades', zorder=5)
                ax_sd.set_title("Supply vs Demand"); ax_sd.set_xlabel("Units (Sorted by Value/Cost)"); ax_sd.set_ylabel("Value / Cost"); ax_sd.legend(fontsize=7); ax_sd.grid(True, which='both', linestyle='--', linewidth=0.5)
                max_y_b = buyer_vals_num[0] if len(buyer_vals_num) > 0 else 0
                max_y_s = seller_vals_num[-1] if len(seller_vals_num) > 0 else 0
                min_y_b = buyer_vals_num[-1] if len(buyer_vals_num) > 0 else 0
                min_y_s = seller_vals_num[0] if len(seller_vals_num) > 0 else 0
                max_y = max(max_y_b, max_y_s) * 1.1; min_y = min(min_y_b, min_y_s) * 0.9
                ax_sd.set_ylim(bottom=max(0, min_y)); ax_sd.set_xlim(left=0.5)
            else: ax_sd.set_title("Supply vs Demand (Data Error/Missing)")

            # --- (3) Individual Submitted Bids/Asks ---
            ax_ba = axes[2]; all_submitted_bids, all_submitted_asks = [], []; buyer_names = sorted([f"B{i}" for i in range(num_buyers)]); seller_names = sorted([f"S{i}" for i in range(num_sellers)])
            try:
                if 'bids_submitted' in df_round.columns: all_submitted_bids = df_round['bids_submitted'].apply(safe_literal_eval).tolist()
                if 'asks_submitted' in df_round.columns: all_submitted_asks = df_round['asks_submitted'].apply(safe_literal_eval).tolist()
            except Exception as e: logger.warning(f"R{rnum}: Could not parse submitted bids/asks for plot 3: {e}")
            if all_submitted_bids:
                cmap_b = plt.cm.get_cmap('Blues', max(2, num_buyers) + 2)
                for b_idx, b_name in enumerate(buyer_names):
                     bids_over_time = [step_bids.get(b_name) if isinstance(step_bids, dict) else None for step_bids in all_submitted_bids]; bids_ot_float = pd.to_numeric(bids_over_time, errors='coerce')
                     if not pd.isna(bids_ot_float).all(): ax_ba.plot(step_indices, bids_ot_float, label=f"{b_name}", color=cmap_b(0.5 + 0.5*b_idx/max(1,num_buyers-1)), alpha=0.7, marker='.', linestyle='', markersize=3)
            if all_submitted_asks:
                cmap_s = plt.cm.get_cmap('Reds', max(2, num_sellers) + 2)
                for s_idx, s_name in enumerate(seller_names):
                     asks_over_time = [step_asks.get(s_name) if isinstance(step_asks, dict) else None for step_asks in all_submitted_asks]; asks_ot_float = pd.to_numeric(asks_over_time, errors='coerce')
                     if not pd.isna(asks_ot_float).all(): ax_ba.plot(step_indices, asks_ot_float, label=f"{s_name}", color=cmap_s(0.5 + 0.5*s_idx/max(1,num_sellers-1)), alpha=0.7, marker='.', linestyle='', markersize=3)
            trade_vals_plot3 = pd.to_numeric(df_round["trade_price"], errors='coerce'); trade_mask_plot3 = ~pd.isna(trade_vals_plot3) # Use pd.isna
            if trade_mask_plot3.any(): ax_ba.scatter(step_indices[trade_mask_plot3], trade_vals_plot3[trade_mask_plot3], label="Trade", color='k', marker='x', s=30, zorder=5)
            ax_ba.set_title("Submitted Bids (Blue) / Asks (Red)"); ax_ba.set_xlabel("Step Index (Across Periods)"); ax_ba.set_ylabel("Price"); ax_ba.grid(True, which='both', linestyle='--', linewidth=0.5)

            # --- Save Figure ---
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            out_path = os.path.join(exp_path, f"round_{rnum:04d}_plot.png"); plt.savefig(out_path, dpi=100); plt.close(fig); plot_count += 1
            if plot_count % 100 == 0: logger.info(f"Generated {plot_count} per-round plots...") # Progress update

        except Exception as e:
            logger.error(f"Failed to generate plot for round {rnum}: {e}", exc_info=True)
            if fig is not None and plt: plt.close(fig) # Ensure figure is closed on error

    logger.info(f"Finished generating {plot_count} per-round plots.")


# --- plot_dqn_behavior_eval function (with fix) ---
def plot_dqn_behavior_eval(dfR, dfLogs, config, exp_path, num_rounds_to_plot=5):
    """
    Generates plots showing the DQN agent's behavior during evaluation rounds.
    Focuses on submitted prices and trades relative to market quotes.
    """
    logger = logging.getLogger('plotting.dqn_eval')
    if not plt_available: logger.warning("Plotting disabled (matplotlib not found or backend error)."); return
    if dfLogs is None or dfLogs.empty: logger.warning("No step logs provided for DQN evaluation behavior plots."); return
    if dfR is None or dfR.empty: logger.warning("No round logs provided for DQN evaluation behavior plots."); return

    # --- Identify DQN Agent ---
    dqn_agent_name = None
    dqn_agent_role = None
    dqn_agent_type = None
    # Find the first agent whose type contains 'dqn'
    for i, buyer_spec in enumerate(config.get('buyers', [])):
        spec_type = buyer_spec.get('type', '').lower()
        if 'dqn' in spec_type:
            dqn_agent_name = f"B{i}"
            dqn_agent_role = 'buyer'
            dqn_agent_type = spec_type
            break
    if not dqn_agent_name:
        for i, seller_spec in enumerate(config.get('sellers', [])):
             spec_type = seller_spec.get('type', '').lower()
             if 'dqn' in spec_type:
                dqn_agent_name = f"S{i}"
                dqn_agent_role = 'seller'
                dqn_agent_type = spec_type
                break

    if not dqn_agent_name:
        logger.warning("Could not find agent with 'dqn' in its type in config. Skipping evaluation plots.")
        return
    logger.info(f"Generating evaluation behavior plots for agent: {dqn_agent_name} ({dqn_agent_role}, type={dqn_agent_type})")

    # --- Select Evaluation Rounds ---
    training_rounds = config.get('num_training_rounds', 0)
    # Ensure 'round' column is numeric
    try: dfR['round'] = pd.to_numeric(dfR['round'])
    except Exception as e: logger.error(f"Could not convert 'round' column to numeric in dfR: {e}"); return
    eval_rounds = dfR[dfR['round'] >= training_rounds]['round'].unique()

    if len(eval_rounds) == 0:
        logger.warning("No evaluation rounds found in data (round >= num_training_rounds).")
        return

    # Plot a sample of evaluation rounds (e.g., the very last ones)
    rounds_to_plot = sorted(eval_rounds)[-num_rounds_to_plot:]
    if not rounds_to_plot: logger.warning("No specific rounds selected for plotting."); return
    logger.info(f"Plotting behavior for evaluation rounds: {rounds_to_plot}")

    # --- Create Output Directory ---
    plot_dir = os.path.join(exp_path, "eval_behavior_plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Convert round column once before loop if possible
    try: dfLogs['round'] = pd.to_numeric(dfLogs['round'])
    except Exception as e: logger.error(f"Could not convert 'round' column to numeric in dfLogs: {e}"); return

    # --- Generate Plot for Each Selected Round ---
    for rnum in rounds_to_plot:
        fig = None # Initialize fig to None
        try:
            # Filter logs for this round
            if 'round' not in dfLogs.columns: logger.error("Missing 'round' column in dfLogs"); continue

            df_round = dfLogs[dfLogs["round"] == rnum].copy()
            if df_round.empty:
                logger.debug(f"No step logs found for evaluation round {rnum}.")
                continue

            # Ensure data is sorted correctly
            sort_cols = ['period', 'step']
            if not all(col in df_round.columns for col in sort_cols): logger.error(f"Missing sort columns {sort_cols} in dfLogs"); continue
            # Convert period/step to numeric for sorting if they aren't already
            try:
                df_round['period'] = pd.to_numeric(df_round['period'])
                df_round['step'] = pd.to_numeric(df_round['step'])
            except Exception as e: logger.error(f"Could not convert period/step to numeric for round {rnum}: {e}"); continue
            df_round.sort_values(by=sort_cols, inplace=True)
            step_indices = np.arange(len(df_round)) # Use a simple index for plotting

            # Extract relevant data series, converting to numeric safely
            market_bid = pd.to_numeric(df_round["current_bid_price"], errors='coerce')
            market_ask = pd.to_numeric(df_round["current_ask_price"], errors='coerce')
            trade_price = pd.to_numeric(df_round["trade_price"], errors='coerce')
            is_trade = df_round["trade_executed"] == 1

            # Extract the DQN agent's specific actions (bids or asks)
            agent_actions_numeric = pd.Series(np.nan, index=df_round.index) # Initialize with NaN
            submitted_col = 'bids_submitted' if dqn_agent_role == 'buyer' else 'asks_submitted'
            if submitted_col in df_round.columns:
                agent_actions = []
                for submissions_str in df_round[submitted_col]:
                    submissions_dict = safe_literal_eval(submissions_str) # Use safe eval
                    agent_actions.append(submissions_dict.get(dqn_agent_name) if isinstance(submissions_dict, dict) else None)
                agent_actions_numeric = pd.to_numeric(agent_actions, errors='coerce')

            # Identify trades involving the DQN agent
            is_agent_buyer = df_round["trade_buyer"] == dqn_agent_name
            is_agent_seller = df_round["trade_seller"] == dqn_agent_name
            agent_trade_mask = is_trade & (is_agent_buyer | is_agent_seller)

            # --- Create Plot ---
            fig, ax = plt.subplots(1, 1, figsize=(18, 7)) # Slightly larger figure
            fig.suptitle(f"DQN Agent '{dqn_agent_name}' Behavior - Evaluation Round {rnum}", fontsize=14)

            # 1. Plot Market Context (Best Bid/Ask)
            ax.plot(step_indices, market_bid, label="Market Bid", color='dimgray', linestyle=':', alpha=0.6, linewidth=1)
            ax.plot(step_indices, market_ask, label="Market Ask", color='darkorange', linestyle=':', alpha=0.6, linewidth=1)

            # 2. Plot DQN Agent's Submitted Quotes
            # <<< FIX HERE: Use pd.isna() >>>
            agent_action_mask = ~pd.isna(agent_actions_numeric)
            action_label = f"{dqn_agent_name} Submitted {'Bid' if dqn_agent_role == 'buyer' else 'Ask'}"
            action_color = 'blue' if dqn_agent_role == 'buyer' else 'red'
            action_marker = '^' if dqn_agent_role == 'buyer' else 'v'
            if agent_action_mask.any(): # Only plot if there are actions
                ax.scatter(step_indices[agent_action_mask], agent_actions_numeric[agent_action_mask],
                           label=action_label, color=action_color, marker=action_marker, s=40, alpha=0.8, zorder=4)

            # 3. Plot Trades Involving the DQN Agent
            trade_label = f"{dqn_agent_name} Trade"
            trade_color = 'green'
            trade_marker = 'o'
            # <<< FIX HERE: Use pd.isna() >>>
            trade_price_valid_mask = ~pd.isna(trade_price)
            combined_agent_trade_mask = agent_trade_mask & trade_price_valid_mask

            if combined_agent_trade_mask.any(): # Only plot if there are trades involving the agent
                ax.scatter(step_indices[combined_agent_trade_mask], trade_price[combined_agent_trade_mask],
                           label=trade_label, color=trade_color, marker=trade_marker, s=60, zorder=5,
                           facecolors='none', edgecolors=trade_color, linewidths=1.5)

            # --- Formatting ---
            ax.set_xlabel("Step Index (Across Periods)")
            ax.set_ylabel("Price")
            ax.legend(fontsize=9, loc='upper right')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)

            # Add period boundaries if multiple periods exist
            num_periods = config.get("num_periods", 1)
            steps_per_period = config.get("num_steps", 25)
            if num_periods > 1:
                for p_boundary in range(1, num_periods):
                    boundary_step_index = p_boundary * steps_per_period - 0.5 # Position between steps
                    ax.axvline(boundary_step_index, color='k', linestyle='--', linewidth=0.7, alpha=0.5)
                # Adjust x-axis ticks to be less dense
                if mticker: ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=20, integer=True))


            # Adjust y-limits slightly based on data range
            # Combine all relevant price series, drop NaNs before finding min/max
            # <<< FIX HERE: Ensure all inputs to concat are Series >>>
            all_valid_price_series = [s for s in [market_bid, market_ask, agent_actions_numeric, trade_price] if isinstance(s, pd.Series)]
            if all_valid_price_series:
                 all_prices = pd.concat(all_valid_price_series).dropna()
                 if not all_prices.empty:
                     min_val = all_prices.min()
                     max_val = all_prices.max()
                     padding = max((max_val - min_val) * 0.05, 1.0) # Add at least 1 unit padding
                     ax.set_ylim(max(0, min_val - padding), max_val + padding) # Ensure y starts >= 0

            # --- Save Plot ---
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
            out_path = os.path.join(plot_dir, f"eval_round_{rnum:05d}_behavior.png") # Pad round number
            try:
                plt.savefig(out_path, dpi=120) # Good resolution for viewing
                logger.debug(f"Saved DQN behavior plot: {out_path}")
            except Exception as e:
                logger.error(f"Error saving DQN behavior plot for round {rnum}: {e}")
            finally:
                plt.close(fig) # Close figure to free memory

        except Exception as e:
            logger.error(f"Failed to generate behavior plot for round {rnum}: {e}", exc_info=True)
            # Ensure figure is closed if an error occurred during processing
            if fig is not None and plt: plt.close(fig)

    logger.info(f"Finished generating DQN evaluation behavior plots in {plot_dir}")


# --- plot_game_summary function (was missing, now included) ---
def plot_game_summary(dfR, exp_path, dfLogs=None):
    """Generates a multi-panel summary plot for the entire game."""
    logger = logging.getLogger('plotting.summary')
    if not plt_available: logger.warning("Plotting disabled."); return
    if dfR is None or dfR.empty: logger.warning("No round data available for game summary plot."); return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"Game Summary: {os.path.basename(exp_path)}", fontsize=14)

    if 'round' not in dfR.columns: logger.error("Missing 'round' column in round data."); plt.close(fig); return
    rounds = dfR["round"].values
    num_rounds_total = len(rounds)
    if num_rounds_total == 0: logger.warning("No rounds in data for summary plot."); plt.close(fig); return

    # --- (0,0) Efficiency & Trade Volume ---
    ax_eff = axes[0, 0]
    ax_eff.plot(rounds, dfR["market_efficiency"], label="Market Eff.", marker='o', markersize=2, linestyle='-', linewidth=1, color='tab:blue', alpha=0.6)
    # Add rolling average for efficiency
    rolling_window_eff = max(10, num_rounds_total // 20)
    dfR_eff_rolling = dfR["market_efficiency"].rolling(window=rolling_window_eff, min_periods=1).mean()
    ax_eff.plot(rounds, dfR_eff_rolling, label=f"Eff. ({rolling_window_eff}-rnd avg)", color='navy', linewidth=1.5)
    ax_eff.set_xlabel("Round"); ax_eff.set_ylabel("Efficiency (0-1)", color='tab:blue'); ax_eff.tick_params(axis='y', labelcolor='tab:blue'); ax_eff.set_title("Market Efficiency & Trade Volume"); ax_eff.grid(True, which='major', linestyle='--', linewidth=0.5); ax_eff.set_ylim(-0.05, 1.05)
    ax_vol = ax_eff.twinx(); ax_vol.plot(rounds, dfR["actual_trades"], label="Actual Trades", marker='.', markersize=2, linestyle=':', linewidth=1, color='tab:green', alpha=0.6); ax_vol.plot(rounds, dfR["eq_q"], label="Eq. Quantity", marker=None, linestyle='--', linewidth=1.5, color='tab:orange'); ax_vol.set_ylabel("Number of Trades", color='tab:green'); ax_vol.tick_params(axis='y', labelcolor='tab:green')
    lines_eff, labels_eff = ax_eff.get_legend_handles_labels(); lines_vol, labels_vol = ax_vol.get_legend_handles_labels(); ax_eff.legend(lines_eff + lines_vol, labels_eff + labels_vol, loc='lower right', fontsize=8)

    # --- (0,1) Surplus Distribution ---
    ax_surplus = axes[0, 1]; buyer_surplus_frac, seller_surplus_frac = [], [];
    for idx, row in dfR.iterrows():
        bot_details = safe_literal_eval(row.get("bot_details"));
        if isinstance(bot_details, list):
            buyer_pft = sum(float(b.get("profit", 0.0)) for b in bot_details if isinstance(b,dict) and b.get("role")=="buyer"); seller_pft = sum(float(s.get("profit", 0.0)) for s in bot_details if isinstance(s,dict) and s.get("role")=="seller"); total_pft = buyer_pft + seller_pft
            if total_pft > 1e-9: buyer_surplus_frac.append(buyer_pft / total_pft); seller_surplus_frac.append(seller_pft / total_pft)
            else: buyer_surplus_frac.append(0.5); seller_surplus_frac.append(0.5)
        else: buyer_surplus_frac.append(0.5); seller_surplus_frac.append(0.5)
    rolling_window_surplus = max(10, num_rounds_total // 20); buyer_roll_avg = pd.Series(buyer_surplus_frac).rolling(window=rolling_window_surplus, min_periods=1).mean(); seller_roll_avg = pd.Series(seller_surplus_frac).rolling(window=rolling_window_surplus, min_periods=1).mean()
    ax_surplus.plot(rounds, buyer_roll_avg, label=f"Buyer Surplus % ({rolling_window_surplus}-rnd avg)", color='blue', linewidth=1.5); ax_surplus.plot(rounds, seller_roll_avg, label=f"Seller Surplus % ({rolling_window_surplus}-rnd avg)", color='red', linewidth=1.5)
    ax_surplus.set_xlabel("Round"); ax_surplus.set_ylabel("Surplus Fraction"); ax_surplus.set_title("Surplus Distribution (Rolling Avg)"); ax_surplus.grid(True, which='both', linestyle='--', linewidth=0.5); ax_surplus.legend(fontsize=8); ax_surplus.set_ylim(-0.1, 1.1)

    # --- (1,0) Price Convergence ---
    ax_price = axes[1, 0]; avg_price = pd.to_numeric(dfR["avg_price"], errors='coerce'); eq_price = pd.to_numeric(dfR["eq_p"], errors='coerce')
    ax_price.plot(rounds, avg_price, label="Avg. Trade Price", marker='.', markersize=2, linestyle='-', linewidth=1, color='green', alpha=0.7); ax_price.plot(rounds, eq_price, label="Eq. Price", marker=None, linestyle='--', linewidth=1.5, color='grey')
    ax_price.set_xlabel("Round"); ax_price.set_ylabel("Price"); ax_price.set_title("Price Convergence"); ax_price.grid(True, which='major', linestyle='--', linewidth=0.5); ax_price.legend(fontsize=8)

    # --- (1,1) Rolling Avg Profit per Agent Strategy/Role ---
    ax_profit = axes[1, 1]; rows = []; agent_keys = set()
    for i, row in dfR.iterrows():
        rnum = row["round"]; agg_dict = safe_literal_eval(row.get("role_strat_perf"));
        if not isinstance(agg_dict, dict): continue
        for key_str, val in agg_dict.items():
             key = safe_literal_eval(key_str);
             if not (isinstance(key, tuple) and len(key) == 2 and isinstance(val, dict)): continue
             role, strat = key; agent_keys.add(key); total_p = float(val.get("profit", 0.0)); count = int(val.get("count", 0)); avg_p = total_p / count if count > 0 else 0.0; rows.append([rnum, role, strat, avg_p])
    if rows:
        dfAgents = pd.DataFrame(rows, columns=["round", "role", "strategy", "avgProfit"]);
        try:
            dfPivot = dfAgents.pivot_table(index="round", columns=["role", "strategy"], values="avgProfit"); rolling_window_profit = max(10, num_rounds_total // 10); dfRolling = dfPivot.rolling(window=rolling_window_profit, min_periods=1).mean()
            cmap_profit = plt.cm.get_cmap('tab10'); color_idx = 0; plotted_lines = []; sorted_agent_keys = sorted(list(agent_keys))
            for role, strat in sorted_agent_keys:
                col_tuple = (role, strat);
                if col_tuple in dfRolling.columns: label_str = f"{role}-{strat}"; line, = ax_profit.plot(dfRolling.index, dfRolling[col_tuple], label=label_str, linewidth=1.5, color=cmap_profit(color_idx % 10)); plotted_lines.append(line); color_idx += 1
            ax_profit.set_xlabel("Round"); ax_profit.set_ylabel(f"Avg Profit ({rolling_window_profit}-rnd roll avg)"); ax_profit.set_title("Agent Average Profit (Rolling Avg)");
            if len(plotted_lines) > 6: ax_profit.legend(handles=plotted_lines, fontsize=7, loc='center left', bbox_to_anchor=(1, 0.5))
            else: ax_profit.legend(handles=plotted_lines, fontsize=8)
            ax_profit.grid(True, which='major', linestyle='--', linewidth=0.5)
        except Exception as e: logger.error(f"Could not pivot/plot rolling profit: {e}", exc_info=True); ax_profit.set_title("Agent Avg Profit (Plotting Error)")
    else: ax_profit.set_title("Agent Avg Profit (No Data)")

    # --- Save Figure ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.92]) # Adjust layout to prevent title overlap
    summary_path = os.path.join(exp_path, "game_summary_plot.png")
    try:
        plt.savefig(summary_path, dpi=150) # Increase DPI for summary plot
        logger.info(f"Saved game summary plot to {summary_path}")
    except Exception as e:
        logger.error(f"Error saving game summary plot: {e}")
    plt.close(fig) # Close figure