# utils.py
import random
import logging
import math
from collections import defaultdict
import numpy as np
import pandas as pd # Ensure pandas is imported
import matplotlib
try:
    # Attempt to use a non-interactive backend suitable for servers/scripts
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    plt_available = True
except ImportError:
    plt = None
    mticker = None
    plt_available = False
    print("Warning: Matplotlib not found or backend error. Plotting disabled.")
except Exception as e:
    plt = None
    mticker = None
    plt_available = False
    print(f"Warning: Error setting up Matplotlib: {e}. Plotting disabled.")


from tabulate import tabulate
import os
import ast # Use ast.literal_eval instead of eval for safety

# Configure plotting style if available
if plt_available:
    try:
        plt.style.use('ggplot')
        # Update parameters for potentially better appearance
        matplotlib.rcParams.update({
            'font.size': 10,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'savefig.facecolor': 'white', # Ensure saved figures also have white background
            'figure.dpi': 100,
            'savefig.dpi': 150, # Higher DPI for saved figures
        })
    except Exception as e:
        print(f"Warning: Could not apply matplotlib style: {e}")

# Get logger for utilities
# Note: Child loggers can be created like logging.getLogger('utils.equilibrium')
logger = logging.getLogger('utils')


# --- Equilibrium Calculation ---
def compute_equilibrium(buyer_vals, seller_costs):
    """ Compute theoretical equilibrium quantity, price range, midpoint, and max surplus. """
    eq_logger = logging.getLogger('utils.equilibrium') # Specific logger
    try:
        # Ensure inputs are numeric and sort them
        b_vals_float = sorted([float(v) for v in buyer_vals if isinstance(v, (int, float, np.number))], reverse=True)
        s_costs_float = sorted([float(c) for c in seller_costs if isinstance(c, (int, float, np.number))])
    except (ValueError, TypeError) as e:
        eq_logger.error(f"Invalid input for equilibrium calculation: {e}. Buyers: {buyer_vals}, Sellers: {seller_costs}")
        return 0, (0.0, 0.0), 0.0, 0.0 # Return defaults

    if not b_vals_float:
        eq_logger.warning("No valid buyer values for equilibrium calculation.")
        # Define behavior when no buyers: Q=0, Price range might depend on seller costs if needed
        low_p = min(s_costs_float) if s_costs_float else 0.0
        return 0, (low_p, low_p), 0.0, 0.0
    if not s_costs_float:
        eq_logger.warning("No valid seller costs for equilibrium calculation.")
        high_p = max(b_vals_float) if b_vals_float else 0.0
        return 0, (high_p, high_p), 0.0, 0.0

    nb = len(b_vals_float)
    ns = len(s_costs_float)
    max_possible_q = min(nb, ns)
    eq_q = 0
    total_surplus = 0.0

    # Determine equilibrium quantity and surplus
    for q in range(1, max_possible_q + 1):
        b_val = b_vals_float[q - 1]
        s_cost = s_costs_float[q - 1]
        if b_val >= s_cost:
            eq_q = q
            total_surplus += (b_val - s_cost)
        else:
            # Stop when the q-th buyer's value is less than the q-th seller's cost
            break

    # Determine equilibrium price range
    if eq_q == 0:
        # No trades possible, price range is between highest buyer value and lowest seller cost
        p_low = s_costs_float[0]
        p_high = b_vals_float[0]
        eq_p_range = (min(p_low, p_high), max(p_low, p_high))
    else:
        # Trades occur, price range determined by the marginal units
        last_included_seller_cost = s_costs_float[eq_q - 1]
        last_included_buyer_value = b_vals_float[eq_q - 1]

        # Value of the first excluded buyer (or -inf if all buyers trade)
        next_buyer_value = b_vals_float[eq_q] if eq_q < nb else -np.inf
        # Cost of the first excluded seller (or +inf if all sellers trade)
        next_seller_cost = s_costs_float[eq_q] if eq_q < ns else np.inf

        # Price must be >= highest cost/value among non-traded units that *could* have traded
        # This defines the lower bound of the equilibrium price range
        eq_p_low = max(last_included_seller_cost, next_buyer_value)
        # Price must be <= lowest value/cost among non-traded units that *could* have traded
        # This defines the upper bound of the equilibrium price range
        eq_p_high = min(last_included_buyer_value, next_seller_cost)

        # Ensure low <= high, adjust if they cross due to discrete values or specific edge cases
        eq_p_range = (min(eq_p_low, eq_p_high), max(eq_p_low, eq_p_high))

    # Calculate midpoint price
    eq_p_mid = float(0.5 * (eq_p_range[0] + eq_p_range[1]))

    # Log potential issues
    if total_surplus < 1e-9 and eq_q > 0:
        eq_logger.warning(f"Equilibrium has Q={eq_q} but near-zero Max Surplus ({total_surplus:.4f}). Price range: {eq_p_range}")

    eq_logger.debug(f"Equilibrium: Q={eq_q}, P_range=({eq_p_range[0]:.2f}, {eq_p_range[1]:.2f}), P_mid={eq_p_mid:.2f}, Max Surplus={total_surplus:.2f}")
    return eq_q, eq_p_mid, total_surplus


# --- SFI Value Generation Functions ---
def generate_sfi_components(gametype, min_price, max_price, value_rng):
    """ Generates SFI R and A components based on gametype. """
    sfi_logger = logging.getLogger('utils.sfi_values')
    sfi_logger.debug(f"Generating SFI components for gametype={gametype}")
    try:
        gt_int = int(round(gametype))
        if not (0 <= gt_int <= 9999): raise ValueError("Gametype out of range 0-9999")
        gt_str = f"{gt_int:04d}" # Pad with leading zeros
        k = [int(digit) for digit in gt_str]
        # R[i] = 3^k[i] - 1, ensuring non-negative R
        R_raw = [(3**ki - 1) for ki in k]
        R = [max(0, r_val) for r_val in R_raw]
        sfi_logger.debug(f"GT {gametype} -> k={k} -> R={R}")
    except Exception as e:
        sfi_logger.error(f"Error processing gametype {gametype}: {e}. Defaulting R=[0,0,0,0].", exc_info=True)
        R = [0, 0, 0, 0]

    R1, _, _, _ = R
    # Generate A based on R1
    A = value_rng.randint(0, R1) if R1 > 0 else 0
    sfi_logger.debug(f"Generated SFI components: R={R}, A={A}")
    return {'R': R, 'A': A}

def calculate_sfi_values_for_participant(p_name, role_l, N, sfi_components, min_p, max_p, value_rng):
    """ Calculates N values/costs for a specific participant based on SFI components. """
    sfi_logger = logging.getLogger('utils.sfi_values')
    R = sfi_components['R']
    A = sfi_components['A']
    _, R2, R3, R4 = R # Unpack R components

    values = []
    # Generate B component once per participant (if R2 > 0)
    B_comp = value_rng.randint(0, R2) if R2 > 0 else 0

    for k in range(1, N + 1): # Generate N values/costs
        # Generate Ck and Djk per token
        Ck = value_rng.randint(0, R3) if R3 > 0 else 0
        Djk = value_rng.randint(0, R4) if R4 > 0 else 0

        # Calculate base value/cost T_jkl
        if role_l == 1: # Buyer (uses B component)
            T_jkl = A + B_comp + Ck + Djk
        elif role_l == 2: # Seller (does not use B component)
            T_jkl = A + Ck + Djk
        else:
            sfi_logger.error(f"Unknown role_l: {role_l} for participant {p_name}. Setting value to 0.")
            T_jkl = 0

        # Clamp the value/cost to market bounds and round to integer
        T_jkl_clamped = int(round(max(min_p, min(max_p, T_jkl))))

        # Log clamping if it occurs (at debug level)
        if abs(T_jkl - T_jkl_clamped) > 1e-6 and sfi_logger.isEnabledFor(logging.DEBUG):
            sfi_logger.debug(f"Value {T_jkl:.2f} clamped/rounded to {T_jkl_clamped} for P:{p_name}, Token:{k}")

        values.append(T_jkl_clamped)

    # Sort values/costs appropriately (buyers descending, sellers ascending) after generation
    if role_l == 1: values.sort(reverse=True)
    else: values.sort()

    sfi_logger.debug(f"Calculated values/costs for P:{p_name} (role {role_l}, N={N}): {values}")
    return values


# --- Analysis Functions ---
def safe_literal_eval(val):
    """ Safely evaluate a string containing a Python literal (list, dict, tuple, etc.). """
    if isinstance(val, (list, dict, tuple, int, float, bool)) or val is None:
        # If it's already a literal type or None, return it directly
        return val
    if isinstance(val, str):
        # Basic check for potentially unsafe constructs (optional, ast is generally safe)
        # if "__" in val or "import" in val or "eval" in val: return None
        try:
            # Attempt to parse the string
            # Handle potential tuple string format like "('buyer', 'zic')"
            processed_val = val
            if val.startswith("(") and val.endswith(")") and "," in val and "'" in val:
                 # Simple heuristic check for tuple-like string
                 try:
                     # More robust parsing could be added if needed
                     pass
                 except: pass # Ignore if formatting is unexpected

            # Use ast.literal_eval for safe evaluation
            return ast.literal_eval(processed_val)
        except (ValueError, SyntaxError, TypeError, MemoryError) as e:
            # Log the error if evaluation fails
            eval_logger = logging.getLogger('utils.safe_eval')
            # Limit log frequency if this happens often? Maybe count occurrences?
            eval_logger.debug(f"safe_literal_eval failed for value '{val}': {e}")
            return None # Return None if evaluation fails
    # Return None for other types (like custom objects)
    return None

def analyze_individual_performance(round_stats):
    """
    Aggregates performance by individual bot instance across rounds.
    Separates buyers and sellers and ranks them by Mean Profit within each role.
    """
    ind_logger = logging.getLogger('analysis.individual')
    if not round_stats: return "No round stats for individual performance analysis."

    # Structure: { (role, strategy, name): [profit1, profit2, ...], ... }
    all_bots_profit = defaultdict(list)
    num_processed_rounds = 0

    # --- Stage 1: Collect profits for each bot ---
    for r_idx, rstat in enumerate(round_stats):
        if not isinstance(rstat, dict): continue # Skip non-dict entries

        bot_details_raw = rstat.get("bot_details")
        if bot_details_raw is None: continue # Skip if details missing

        bot_details = safe_literal_eval(bot_details_raw)
        if not isinstance(bot_details, list) or not bot_details: continue # Skip if not valid list

        num_processed_rounds += 1
        for b_idx, b in enumerate(bot_details):
            if not isinstance(b, dict): continue # Skip non-dict bot entries

            role = b.get("role")
            strategy = b.get("strategy")
            name = b.get("name")
            profit_raw = b.get("profit")

            # Validate key components
            if None in [role, strategy, name] or not all(isinstance(k, str) for k in [role, strategy, name]):
                ind_logger.debug(f"R{r_idx}: Skipping bot entry with invalid key components: {b}")
                continue
            key = (role, strategy, name)

            # Validate and convert profit
            try:
                profit_val = float(profit_raw if profit_raw is not None else 0.0)
                all_bots_profit[key].append(profit_val)
            except (ValueError, TypeError):
                ind_logger.warning(f"R{r_idx}: Invalid profit '{profit_raw}' for bot {key}. Skipping profit entry.")

    if not all_bots_profit:
        return f"No valid bot details found in {num_processed_rounds} processed rounds."

    # --- Stage 2: Calculate aggregate stats and separate by role ---
    buyer_results = []
    seller_results = []

    for key, profit_list in all_bots_profit.items():
        role, strategy, name = key
        arr = np.array(profit_list)
        if len(arr) == 0: continue # Skip if no valid profit entries

        # Use nan-safe calculations
        avg = np.nanmean(arr)
        std = np.nanstd(arr)
        min_p = np.nanmin(arr)
        med = np.nanmedian(arr)
        max_p = np.nanmax(arr)

        result_dict = {
            'role': role,
            'strategy': strategy,
            'name': name,
            'mean_profit': avg,
            'std_profit': std,
            'min_profit': min_p,
            'median_profit': med,
            'max_profit': max_p
        }

        if role == 'buyer':
            buyer_results.append(result_dict)
        elif role == 'seller':
            seller_results.append(result_dict)
        else:
            ind_logger.warning(f"Unknown role '{role}' found for bot {key}. Ignoring.")

    # --- Stage 3: Sort results by Mean Profit (descending) ---
    buyers_sorted = sorted(buyer_results, key=lambda x: x['mean_profit'], reverse=True)
    sellers_sorted = sorted(seller_results, key=lambda x: x['mean_profit'], reverse=True)

    # --- Stage 4: Format tables ---
    headers = ["Rank", "Strategy", "BotName", "MeanProfit", "StdProfit", "MinProfit", "MedianProfit", "MaxProfit"]
    buyer_table_rows = []
    seller_table_rows = []

    for rank, res in enumerate(buyers_sorted, 1):
        buyer_table_rows.append([
            rank, # Add rank column
            res['strategy'], res['name'],
            f"{res['mean_profit']:.2f}", f"{res['std_profit']:.2f}",
            f"{res['min_profit']:.2f}", f"{res['median_profit']:.2f}", f"{res['max_profit']:.2f}"
        ])

    for rank, res in enumerate(sellers_sorted, 1):
        seller_table_rows.append([
            rank, # Add rank column
            res['strategy'], res['name'],
            f"{res['mean_profit']:.2f}", f"{res['std_profit']:.2f}",
            f"{res['min_profit']:.2f}", f"{res['median_profit']:.2f}", f"{res['max_profit']:.2f}"
        ])

    title = f"\n=== INDIVIDUAL BOT PERFORMANCE (Aggregated over {num_processed_rounds} Rounds) ==="
    buyer_title = "\n--- Buyers (Ranked by Mean Profit) ---"
    seller_title = "\n--- Sellers (Ranked by Mean Profit) ---"

    # Use right alignment for numeric columns in tabulate
    buyer_table = tabulate(buyer_table_rows, headers=headers, tablefmt='pretty', floatfmt='.2f', numalign="right") if buyer_table_rows else "No buyer data."
    seller_table = tabulate(seller_table_rows, headers=headers, tablefmt='pretty', floatfmt='.2f', numalign="right") if seller_table_rows else "No seller data."

    # Combine titles and tables
    final_output = f"{title}\n{buyer_title}\n{buyer_table}\n{seller_title}\n{seller_table}"

    return final_output


def analyze_market_performance(round_stats):
    """ Aggregates market performance including standard deviations. """
    mkt_logger = logging.getLogger('analysis.market')
    if not round_stats: return "No round stats for market performance analysis."

    # Lists to store per-round metrics
    effs_raw, effs_clamped = [], []
    price_diffs, quant_diffs = [], []
    buyer_surplus_fracs, seller_surplus_fracs = [], []
    avg_buyer_profits_per_round, avg_seller_profits_per_round = [], []

    eff_issues, neg_eff_rounds, num_processed_rounds = 0, 0, 0

    for r_idx, rstat in enumerate(round_stats):
        if not isinstance(rstat, dict): continue # Skip non-dict entries
        num_processed_rounds += 1

        # --- Efficiency ---
        eff_raw = rstat.get('market_efficiency')
        eff = np.nan # Default to NaN
        try:
             if eff_raw is None: raise ValueError("Missing efficiency")
             eff = float(eff_raw)
             effs_raw.append(eff) # Store raw value
             # Check for issues and clamp for aggregate stats
             is_issue = False
             # Use a tolerance for checking > 1.0 due to potential floating point issues
             if eff > 1.0 + 1e-6: is_issue = True
             if eff < -1e-9: # Check for negative values (allowing for near-zero float issues)
                  neg_eff_rounds +=1
                  is_issue = True # Consider negative efficiency an issue
             if is_issue: eff_issues += 1
             eff_clamped_val = np.clip(eff, 0.0, 1.0) # Clamp for avg/std calculation
             effs_clamped.append(eff_clamped_val)
        except (ValueError, TypeError, AttributeError) as e:
             mkt_logger.warning(f"R{r_idx}: Invalid or missing efficiency value '{eff_raw}': {e}. Using NaN.")
             effs_raw.append(np.nan)
             effs_clamped.append(np.nan)

        # --- Price/Quantity Deviation ---
        adp = rstat.get('abs_diff_price'); adq = rstat.get('abs_diff_quantity')
        try: price_diffs.append(float(adp) if adp is not None else np.nan)
        except (ValueError, TypeError): price_diffs.append(np.nan)
        try: quant_diffs.append(float(adq) if adq is not None else np.nan)
        except (ValueError, TypeError): quant_diffs.append(np.nan)

        # --- Surplus Split & Avg Profits ---
        role_perf_raw = rstat.get("role_strat_perf")
        round_total_b_profit, round_b_count = 0.0, 0
        round_total_s_profit, round_s_count = 0.0, 0
        valid_perf_data = False
        round_total_profit_from_roles = 0.0 # Recalculate from roles for consistency check

        role_perf = safe_literal_eval(role_perf_raw)
        if isinstance(role_perf, dict):
             valid_perf_data = True
             try:
                 for key_repr, perf_data in role_perf.items():
                     # key_repr should be like "('buyer', 'zic')"
                     key_tuple = safe_literal_eval(key_repr)
                     if not (isinstance(key_tuple, tuple) and len(key_tuple)==2 and isinstance(perf_data, dict)):
                          mkt_logger.warning(f"R{r_idx}: Invalid key or data format in role_strat_perf: {key_repr}, {perf_data}")
                          valid_perf_data = False; break # Stop processing this round's role perf

                     role, _ = key_tuple
                     profit = float(perf_data.get("profit", 0.0))
                     count = int(perf_data.get("count", 0))
                     if count < 0: raise ValueError("Negative count")

                     round_total_profit_from_roles += profit
                     if role == 'buyer': round_total_b_profit += profit; round_b_count += count
                     elif role == 'seller': round_total_s_profit += profit; round_s_count += count
                     else: mkt_logger.warning(f"R{r_idx}: Unknown role '{role}' in role_strat_perf key: {key_tuple}")

             except (ValueError, TypeError, KeyError) as e: # Catch potential key errors too
                  mkt_logger.warning(f"R{r_idx}: Error processing role_strat_perf data '{role_perf_raw}': {e}")
                  valid_perf_data = False

        # Calculate fractions and averages if data was valid
        if valid_perf_data and (round_b_count + round_s_count > 0):
             tot_pft_roles = round_total_b_profit + round_total_s_profit
             # Use tot_pft_roles for fraction calculation if valid
             # Handle zero total profit case for fractions
             buyer_frac = round_total_b_profit / tot_pft_roles if abs(tot_pft_roles) > 1e-9 else 0.5
             seller_frac = round_total_s_profit / tot_pft_roles if abs(tot_pft_roles) > 1e-9 else 0.5
             buyer_surplus_fracs.append(buyer_frac)
             seller_surplus_fracs.append(seller_frac)
             avg_buyer_profits_per_round.append(round_total_b_profit / round_b_count if round_b_count > 0 else 0.0)
             avg_seller_profits_per_round.append(round_total_s_profit / round_s_count if round_s_count > 0 else 0.0)
        else:
             # Append NaN if role data was missing or invalid
             buyer_surplus_fracs.append(np.nan); seller_surplus_fracs.append(np.nan)
             avg_buyer_profits_per_round.append(np.nan); avg_seller_profits_per_round.append(np.nan)
             # Log warning only once if consistently missing (avoid flooding)
             if num_processed_rounds == 1 or r_idx % 100 == 0: # Log first time and periodically
                  mkt_logger.warning(f"R{r_idx}: Could not calculate surplus split/avg profits due to missing/invalid role_strat_perf data: {role_perf_raw}")

    # --- Aggregate Results (Mean and Std Dev using nan-safe versions) ---
    if num_processed_rounds == 0:
         return "Market analysis failed: No rounds processed."

    # Log the efficiency issue summary
    if eff_issues > 0:
         mkt_logger.warning(f"Market Analysis: {eff_issues}/{num_processed_rounds} rounds had efficiency outside [0, 1] (Negative: {neg_eff_rounds}). Clamped Efficiency used for avg/std calculation.")

    # Calculate aggregate stats
    avg_eff, std_eff = (np.nanmean(effs_clamped), np.nanstd(effs_clamped)) if effs_clamped else (np.nan, np.nan)
    avg_p_diff, std_p_diff = (np.nanmean(price_diffs), np.nanstd(price_diffs)) if price_diffs else (np.nan, np.nan)
    avg_q_diff, std_q_diff = (np.nanmean(quant_diffs), np.nanstd(quant_diffs)) if quant_diffs else (np.nan, np.nan)
    avg_b_surplus, std_b_surplus = (np.nanmean(buyer_surplus_fracs), np.nanstd(buyer_surplus_fracs)) if buyer_surplus_fracs else (np.nan, np.nan)
    avg_s_surplus, std_s_surplus = (np.nanmean(seller_surplus_fracs), np.nanstd(seller_surplus_fracs)) if seller_surplus_fracs else (np.nan, np.nan)
    avg_b_prof, std_b_prof = (np.nanmean(avg_buyer_profits_per_round), np.nanstd(avg_buyer_profits_per_round)) if avg_buyer_profits_per_round else (np.nan, np.nan)
    avg_s_prof, std_s_prof = (np.nanmean(avg_seller_profits_per_round), np.nanstd(avg_seller_profits_per_round)) if avg_seller_profits_per_round else (np.nan, np.nan)

    # --- Format Output Strings with (Std Dev) ---
    # Helper to format, handling potential NaN results gracefully
    def format_mean_std(mean_val, std_val, precision, is_percent=False):
        if np.isnan(mean_val) or np.isnan(std_val):
             return "NaN (NaN)" # Indicate missing data clearly
        unit = "%" if is_percent else ""
        scale = 100.0 if is_percent else 1.0
        # Format: MeanUnit (StdDev)
        return f"{mean_val * scale:.{precision}f}{unit} ({std_val * scale:.{precision}f})"

    # Format results using the helper
    eff_str = format_mean_std(avg_eff, std_eff, 2) # Efficiency clamped: 2 decimals
    b_prof_str = format_mean_std(avg_b_prof, std_b_prof, 0) # Avg Profit: 0 decimals
    s_prof_str = format_mean_std(avg_s_prof, std_s_prof, 0) # Avg Profit: 0 decimals
    b_surplus_str = format_mean_std(avg_b_surplus, std_b_surplus, 1, is_percent=True) # Surplus %: 1 decimal
    s_surplus_str = format_mean_std(avg_s_surplus, std_s_surplus, 1, is_percent=True) # Surplus %: 1 decimal
    p_dev_str = format_mean_std(avg_p_diff, std_p_diff, 1) # Price Dev: 1 decimal
    q_dev_str = format_mean_std(avg_q_diff, std_q_diff, 1) # Quant Dev: 1 decimal

    # --- Create Table ---
    headers = ["Market Eff", "AvgBuyerProfit", "AvgSellerProfit",
               "BuyerSurplus%", "SellerSurplus%", "AvgPriceDev", "AvgQuantDev"]
    rows = [[eff_str, b_prof_str, s_prof_str,
             b_surplus_str, s_surplus_str,
             p_dev_str, q_dev_str]]

    title = f"\n=== MARKET PERFORMANCE (Mean (StdDev) over {num_processed_rounds} Rounds) ==="
    # Use right alignment for numeric data
    table = tabulate(rows, headers=headers, tablefmt="pretty", numalign="right")
    if eff_issues > 0:
        table += f"\nNote: Efficiency clamped to [0, 1] for {eff_issues} rounds for averaging."

    return f"{title}\n{table}"


def analyze_strategy_tournament(round_stats):
    """
    Analyzes performance aggregated by STRATEGY, ranking them like a tournament.
    Calculates mean/std for profit and rank achieved within rounds.
    """
    tourn_logger = logging.getLogger('analysis.tournament')
    if not round_stats:
        return "No round stats for tournament analysis."

    # Dictionary to store lists of profits and ranks for each strategy
    # Structure: { strategy_name: {'profits': [...], 'ranks': [...] } }
    strategy_performance = defaultdict(lambda: {'profits': [], 'ranks': []})
    num_processed_rounds = 0
    total_agents_processed = 0

    for r_idx, rstat in enumerate(round_stats):
        if not isinstance(rstat, dict):
            tourn_logger.warning(f"Round {r_idx}: Skipping invalid round stat entry (not a dict).")
            continue

        bot_details_raw = rstat.get("bot_details")
        if bot_details_raw is None:
            tourn_logger.warning(f"Round {r_idx}: Skipping round, 'bot_details' missing.")
            continue

        bot_details = safe_literal_eval(bot_details_raw) # Use the safe eval function

        if not isinstance(bot_details, list) or not bot_details:
            tourn_logger.warning(f"Round {r_idx}: Skipping round, 'bot_details' is not a valid list or is empty.")
            continue

        num_processed_rounds += 1
        round_bots = [] # Store {'strategy': s, 'profit': p, 'name': n} for this round
        valid_bots_in_round = True
        for b_idx, b in enumerate(bot_details):
            if not isinstance(b, dict):
                tourn_logger.warning(f"Round {r_idx}, Bot {b_idx}: Skipping invalid bot detail entry (not a dict).")
                valid_bots_in_round = False
                continue # Skip this bot entry

            strategy = b.get("strategy")
            profit_raw = b.get("profit")
            name = b.get("name", f"Unknown_R{r_idx}_B{b_idx}") # Use round/bot index if name missing

            if strategy is None:
                tourn_logger.warning(f"Round {r_idx}, Bot {name}: Skipping bot, missing 'strategy'.")
                valid_bots_in_round = False
                continue
            try:
                # Ensure profit is float, default to 0.0 if None
                profit = float(profit_raw if profit_raw is not None else 0.0)
                round_bots.append({'strategy': strategy, 'profit': profit, 'name': name})
                total_agents_processed += 1
            except (ValueError, TypeError):
                tourn_logger.warning(f"Round {r_idx}, Bot {name}: Skipping bot, invalid profit value '{profit_raw}'.")
                valid_bots_in_round = False
                continue

        # Only perform ranking if the round had valid bot entries
        if not valid_bots_in_round or not round_bots:
            tourn_logger.warning(f"Round {r_idx}: Skipping rank calculation due to invalid bot entries or no valid bots.")
            continue

        # --- Rank bots within this round using pandas ---
        try:
            df_round = pd.DataFrame(round_bots)
            # Rank based on profit (higher profit = lower rank number)
            # 'average' method handles ties by assigning the average rank
            df_round['rank'] = df_round['profit'].rank(method='average', ascending=False)

            # --- Aggregate results per strategy for this round ---
            for _, row in df_round.iterrows():
                strat = row['strategy']
                # Append profit and rank for this agent in this round
                strategy_performance[strat]['profits'].append(row['profit'])
                strategy_performance[strat]['ranks'].append(row['rank'])
        except Exception as e:
             tourn_logger.error(f"Round {r_idx}: Error during ranking or aggregation: {e}", exc_info=True)
             # Continue to next round if possible

    # --- Check if any data was aggregated ---
    if not strategy_performance:
        return f"No valid strategy performance data found across {num_processed_rounds} processed rounds (Total agents processed: {total_agents_processed})."

    # --- Calculate overall stats per strategy ---
    results = []
    for strategy, data in strategy_performance.items():
        profits = np.array(data['profits'])
        ranks = np.array(data['ranks'])

        # Ensure there's data before calculating stats
        if len(profits) == 0 or len(ranks) == 0:
             tourn_logger.warning(f"Strategy '{strategy}' had no valid profit/rank entries. Skipping.")
             continue

        mean_profit = np.nanmean(profits)
        std_profit = np.nanstd(profits)
        mean_rank = np.nanmean(ranks)
        std_rank = np.nanstd(ranks)
        count = len(profits) # Number of agent-rounds for this strategy

        results.append({
            'Strategy': strategy,
            'MeanProfit': mean_profit,
            'StdProfit': std_profit,
            'MeanRank': mean_rank,
            'StdRank': std_rank,
            'Count': count # Total observations (agent-rounds)
        })

    if not results:
        return f"Could not compute final stats for any strategy across {num_processed_rounds} processed rounds."

    # --- Sort results by Mean Rank (ascending - lower rank is better) ---
    results_sorted = sorted(results, key=lambda x: x['MeanRank'])

    # --- Format for Tabulate ---
    table_rows = []
    for res in results_sorted:
        table_rows.append([
            res['Strategy'],
            f"{res['MeanProfit']:.2f}",
            f"({res['StdProfit']:.2f})", # Std Dev in parentheses
            f"{res['MeanRank']:.2f}",
            f"({res['StdRank']:.2f})",  # Std Dev in parentheses
            res['Count']
        ])

    # Adjusted headers for clarity with Std Dev in separate conceptual column
    headers = ["Strategy", "Mean Profit", "(Std Dev)", "Mean Rank", "(Std Dev)", "Agent-Rounds"]
    title = f"\n=== STRATEGY TOURNAMENT RANKING (Aggregated over {num_processed_rounds} Rounds) ==="
    # Use right alignment for numeric columns
    table = tabulate(table_rows, headers=headers, tablefmt="pretty", stralign="left", numalign="right")

    return f"{title}\n{table}"


# --- Plotting Functions ---
# Basic placeholders - replace with your actual plotting logic if needed

def plot_per_round(round_stats_list, exp_path, dfLogs=None, generate_plots=True):
    """ Placeholder for per-round plotting. """
    plot_logger = logging.getLogger('plotting.round')
    if not plt_available: plot_logger.warning("Plotting disabled (matplotlib unavailable)."); return
    if not generate_plots: plot_logger.info("Skipping per-round plots (disabled by config)."); return
    # --- Add actual plotting logic here ---
    plot_logger.debug("plot_per_round called (plotting logic omitted).")
    # Example: Plot price convergence for a few rounds using dfLogs
    # Check if dfLogs is provided and not empty
    # Select a few round numbers from round_stats_list
    # Filter dfLogs for those rounds
    # Create plots of price vs step for each selected round
    pass

def plot_rl_behavior_eval(dfR, dfLogs, config, exp_path, num_rounds_to_plot=5):
    """ Placeholder for plotting RL agent behavior during evaluation. """
    plot_logger = logging.getLogger('plotting.rl_eval')
    if not plt_available: plot_logger.warning("Plotting disabled (matplotlib unavailable)."); return
    if dfR is None or dfR.empty or dfLogs is None or dfLogs.empty:
         plot_logger.warning("Missing data for RL behavior plots.")
         return
    # --- Add actual plotting logic here ---
    plot_logger.debug("plot_rl_behavior_eval called (plotting logic omitted).")
    # Example: Identify RL agents from config/dfR
    # Select a few evaluation rounds
    # Filter dfLogs for RL agents in selected rounds
    # Plot submitted bids/asks, accepted trades vs time step
    pass

def plot_ppo_training_curves(rl_training_logs, exp_path):
    """ Plots PPO training metrics (loss, entropy, etc.) over rounds. """
    plot_logger = logging.getLogger('plotting.ppo_train')
    if not plt_available: plot_logger.warning("Plotting disabled (matplotlib unavailable)."); return
    if not rl_training_logs: plot_logger.info("No RL training logs to plot."); return

    try:
        df_train = pd.DataFrame(rl_training_logs)
        if df_train.empty: plot_logger.info("RL training log DataFrame is empty."); return

        # Ensure required columns exist
        required_cols = ['round', 'avg_policy_loss', 'avg_value_loss', 'avg_entropy']
        if not all(col in df_train.columns for col in required_cols):
             # Check for alternative names if needed, or log specific missing columns
             missing = [col for col in required_cols if col not in df_train.columns]
             plot_logger.warning(f"RL training logs missing required columns {missing}. Cannot plot curves.")
             return

        # Aggregate stats per round (average if multiple agents logged stats for the same round)
        df_grouped = df_train.groupby('round').agg(
            policy_loss = ('avg_policy_loss', 'mean'),
            value_loss = ('avg_value_loss', 'mean'),
            entropy = ('avg_entropy', 'mean'),
            # Use get() with default None for optional columns
            approx_kl = ('avg_approx_kl', lambda x: np.nanmean(x.astype(float))), # Handle potential NaNs
            clip_frac = ('avg_clip_frac', lambda x: np.nanmean(x.astype(float)))
        ).reset_index()

        num_plots = 3 # Start with 3 base plots
        if 'approx_kl' in df_grouped.columns: num_plots +=1
        if 'clip_frac' in df_grouped.columns: num_plots +=1

        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots), sharex=True)
        # Use experiment name in title (extract from path)
        exp_name_title = os.path.basename(os.path.normpath(exp_path))
        fig.suptitle(f'PPO Training Curves ({exp_name_title})')
        ax_idx = 0

        # Plot Policy Loss
        axes[ax_idx].plot(df_grouped['round'], df_grouped['policy_loss'], label='Policy Loss', alpha=0.8, linewidth=1.5)
        axes[ax_idx].set_ylabel('Policy Loss')
        axes[ax_idx].grid(True, linestyle='--', alpha=0.6)
        axes[ax_idx].legend()
        ax_idx += 1

        # Plot Value Loss
        axes[ax_idx].plot(df_grouped['round'], df_grouped['value_loss'], label='Value Loss', color='orange', alpha=0.8, linewidth=1.5)
        axes[ax_idx].set_ylabel('Value Loss')
        axes[ax_idx].grid(True, linestyle='--', alpha=0.6)
        axes[ax_idx].legend()
        ax_idx += 1

        # Plot Entropy
        axes[ax_idx].plot(df_grouped['round'], df_grouped['entropy'], label='Entropy', color='green', alpha=0.8, linewidth=1.5)
        axes[ax_idx].set_ylabel('Policy Entropy')
        axes[ax_idx].grid(True, linestyle='--', alpha=0.6)
        axes[ax_idx].legend()
        ax_idx += 1

        # Plot Approx KL (optional)
        if 'approx_kl' in df_grouped.columns:
             axes[ax_idx].plot(df_grouped['round'], df_grouped['approx_kl'], label='Approx KL', color='red', alpha=0.8, linewidth=1.5)
             axes[ax_idx].set_ylabel('Approx KL Divergence')
             axes[ax_idx].grid(True, linestyle='--', alpha=0.6)
             axes[ax_idx].legend()
             ax_idx += 1

        # Plot Clip Fraction (optional)
        if 'clip_frac' in df_grouped.columns:
             axes[ax_idx].plot(df_grouped['round'], df_grouped['clip_frac'], label='Clip Fraction', color='purple', alpha=0.8, linewidth=1.5)
             axes[ax_idx].set_ylabel('Clip Fraction')
             axes[ax_idx].grid(True, linestyle='--', alpha=0.6)
             axes[ax_idx].legend()
             ax_idx += 1

        # Set x-axis label only on the bottom plot
        axes[-1].set_xlabel('Training Round')

        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent title overlap
        save_path = os.path.join(exp_path, "ppo_training_curves.png")
        try:
            plt.savefig(save_path)
            plot_logger.info(f"Saved PPO training curves plot to {save_path}")
        except Exception as e:
            plot_logger.error(f"Error saving PPO training plot: {e}")
        plt.close(fig) # Close the figure to free memory

    except Exception as e:
        plot_logger.error(f"Error generating PPO training curves plot: {e}", exc_info=True)


def plot_game_summary(dfR, exp_path, dfLogs=None):
    """ Plots overall game summary metrics (e.g., efficiency over rounds). """
    plot_logger = logging.getLogger('plotting.summary')
    if not plt_available: plot_logger.warning("Plotting disabled (matplotlib unavailable)."); return
    if dfR is None or dfR.empty: plot_logger.warning("No round data (dfR) to plot game summary."); return

    try:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        # Use experiment name in title
        exp_name_title = os.path.basename(os.path.normpath(exp_path))
        fig.suptitle(f'Market Summary ({exp_name_title})')

        # --- Plot Efficiency ---
        if 'market_efficiency' in dfR.columns:
             # Clamp efficiency for plotting if values are outside [0,1]
             eff_plot = np.clip(dfR['market_efficiency'].astype(float).fillna(0), 0, 1) # Ensure float before clip
             axes[0].plot(dfR['round'], eff_plot * 100, label='Market Efficiency', alpha=0.7, linewidth=1)
             # Optional: Add a rolling average
             window_size = max(1, min(50, len(dfR)//10)) # Dynamic window size, max 50
             if len(dfR) >= window_size:
                 try: # Rolling can fail on very short series or all NaNs
                     rolling_eff = eff_plot.rolling(window=window_size, center=True, min_periods=1).mean()
                     axes[0].plot(dfR['round'], rolling_eff * 100, label=f'{window_size}-Round Avg Eff', color='red', linestyle='--', linewidth=1.5)
                 except Exception as roll_e:
                     plot_logger.warning(f"Could not calculate rolling efficiency: {roll_e}")
             axes[0].set_ylabel('Efficiency (%)')
             axes[0].set_ylim(-5, 105) # Allow slight margin outside 0-100
             axes[0].grid(True, linestyle='--', alpha=0.6)
             axes[0].legend()
        else:
             axes[0].text(0.5, 0.5, 'Market Efficiency data not found', horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)
             axes[0].set_ylabel('Efficiency (%)')

        # --- Plot Price Deviation ---
        if 'abs_diff_price' in dfR.columns and 'eq_p' in dfR.columns:
             # Calculate relative price deviation (%) if possible
             # Replace 0 eq_p with NaN to avoid division by zero, then fill resulting NaN deviation with 0
             eq_p_safe = dfR['eq_p'].astype(float).replace(0, np.nan)
             rel_price_dev = (dfR['abs_diff_price'].astype(float) / eq_p_safe * 100).fillna(0)
             axes[1].plot(dfR['round'], rel_price_dev, label='Avg Price Deviation', alpha=0.7, linewidth=1)
             if len(dfR) >= window_size:
                 try:
                     rolling_dev = rel_price_dev.rolling(window=window_size, center=True, min_periods=1).mean()
                     axes[1].plot(dfR['round'], rolling_dev, label=f'{window_size}-Round Avg Dev', color='red', linestyle='--', linewidth=1.5)
                 except Exception as roll_e:
                     plot_logger.warning(f"Could not calculate rolling price deviation: {roll_e}")
             axes[1].set_ylabel('Avg Price Dev from Eq (%)')
             axes[1].grid(True, linestyle='--', alpha=0.6)
             axes[1].legend()
             # axes[1].set_yscale('log') # Optional: log scale if deviation varies greatly, needs careful handling of zeros
             axes[1].set_ylim(bottom=-1) # Allow slight negative if needed, but focus on positive deviations
        else:
             axes[1].text(0.5, 0.5, 'Price Deviation data not found', horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)
             axes[1].set_ylabel('Avg Price Dev from Eq (%)')

        axes[-1].set_xlabel('Round') # Set x-label only on bottom plot

        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout
        save_path = os.path.join(exp_path, "market_summary_plot.png")
        try:
            plt.savefig(save_path)
            plot_logger.info(f"Saved market summary plot to {save_path}")
        except Exception as e:
            plot_logger.error(f"Error saving market summary plot: {e}")
        plt.close(fig) # Close the figure to free memory

    except Exception as e:
        plot_logger.error(f"Error generating game summary plot: {e}", exc_info=True)