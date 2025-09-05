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
    # Suppress matplotlib font manager debug messages
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
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
    eq_logger.debug(f"Computing equilibrium...") # Log start

    try:
        # Ensure inputs are numeric and sort them
        b_vals_float = sorted([float(v) for v in buyer_vals if isinstance(v, (int, float, np.number))], reverse=True)
        s_costs_float = sorted([float(c) for c in seller_costs if isinstance(c, (int, float, np.number))])
        eq_logger.debug(f"  Input Buyer Vals (sorted): {b_vals_float}")
        eq_logger.debug(f"  Input Seller Costs (sorted): {s_costs_float}")
    except (ValueError, TypeError) as e:
        eq_logger.error(f"Invalid input for equilibrium calculation: {e}. Buyers: {buyer_vals}, Sellers: {seller_costs}")
        return 0, (0.0, 0.0), 0.0, 0.0 # Return defaults

    if not b_vals_float:
        eq_logger.warning("No valid buyer values for equilibrium calculation.")
        low_p = min(s_costs_float) if s_costs_float else 0.0
        return 0, (low_p, low_p), low_p, 0.0 # Return default price, 0 surplus
    if not s_costs_float:
        eq_logger.warning("No valid seller costs for equilibrium calculation.")
        high_p = max(b_vals_float) if b_vals_float else 0.0
        return 0, (high_p, high_p), high_p, 0.0 # Return default price, 0 surplus

    nb = len(b_vals_float)
    ns = len(s_costs_float)
    max_possible_q = min(nb, ns)
    eq_q = 0
    total_surplus = 0.0
    eq_logger.debug(f"  Max possible Q = min({nb}, {ns}) = {max_possible_q}") # Log max Q

    # Determine equilibrium quantity and surplus
    for q in range(1, max_possible_q + 1):
        b_val = b_vals_float[q - 1]
        s_cost = s_costs_float[q - 1]
        if b_val >= s_cost:
            eq_q = q
            surplus_gain = b_val - s_cost
            total_surplus += surplus_gain
            eq_logger.debug(f"    q={q}: BuyerVal={b_val:.2f} >= SellerCost={s_cost:.2f}. Gain={surplus_gain:.2f}. Cumulative Surplus={total_surplus:.2f}. EqQ now {eq_q}.")
        else:
            eq_logger.debug(f"    q={q}: BuyerVal={b_val:.2f} < SellerCost={s_cost:.2f}. Stopping Q search.")
            break

    # Determine equilibrium price range
    # Based on logic from https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1510416 (Page 7)
    eq_p_range = (0.0, 0.0) # Default
    if eq_q == 0: # No trade possible
        # Price range is between highest buyer value and lowest seller cost
        p_low = s_costs_float[0]
        p_high = b_vals_float[0]
        eq_p_range = (min(p_low, p_high), max(p_low, p_high))
    elif eq_q < max_possible_q: # Normal case with intra-marginal and extra-marginal units
        # Lowest price is max(Cost[eq_q], Value[eq_q+1])
        # Highest price is min(Value[eq_q], Cost[eq_q+1])
        last_included_seller_cost = s_costs_float[eq_q - 1]
        last_included_buyer_value = b_vals_float[eq_q - 1]
        next_seller_cost = s_costs_float[eq_q]
        next_buyer_value = b_vals_float[eq_q]
        eq_p_low = max(last_included_seller_cost, next_buyer_value)
        eq_p_high = min(last_included_buyer_value, next_seller_cost)
        eq_p_range = (min(eq_p_low, eq_p_high), max(eq_p_low, eq_p_high)) # Order them correctly
    else: # eq_q == max_possible_q (All units of shorter side trade)
        last_included_seller_cost = s_costs_float[eq_q - 1]
        last_included_buyer_value = b_vals_float[eq_q - 1]
        if nb == ns: # Equal number, both exhausted
             eq_p_range = (last_included_seller_cost, last_included_buyer_value)
        elif nb > ns: # More buyers than sellers, sellers exhausted
             next_buyer_value = b_vals_float[eq_q] # ns == eq_q
             eq_p_low = max(last_included_seller_cost, next_buyer_value)
             eq_p_high = last_included_buyer_value
             eq_p_range = (min(eq_p_low, eq_p_high), max(eq_p_low, eq_p_high))
        else: # ns > nb, Buyers exhausted
             next_seller_cost = s_costs_float[eq_q] # nb == eq_q
             eq_p_low = last_included_seller_cost
             eq_p_high = min(last_included_buyer_value, next_seller_cost)
             eq_p_range = (min(eq_p_low, eq_p_high), max(eq_p_low, eq_p_high))

    # Calculate midpoint price
    eq_p_mid = float(0.5 * (eq_p_range[0] + eq_p_range[1]))

    if abs(total_surplus) < 1e-9 and eq_q > 0:
        eq_logger.warning(f"Equilibrium has Q={eq_q} but near-zero Max Surplus ({total_surplus:.4f}). Price range: {eq_p_range}")

    eq_logger.debug(f"  Equilibrium Result: Q={eq_q}, P_range=({eq_p_range[0]:.2f}, {eq_p_range[1]:.2f}), P_mid={eq_p_mid:.2f}, Calculated Max Surplus={total_surplus:.2f}")
    # Return Q, Midpoint Price, and Max Theoretical Surplus
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
        return val
    if isinstance(val, str):
        try:
            # Basic check for common safe types - extend if needed
            if not (val.startswith('[') or val.startswith('{') or val.startswith('(')):
                 # If it doesn't look like a container, try simple types first
                 try: return int(val)
                 except ValueError: pass
                 try: return float(val)
                 except ValueError: pass
                 if val.lower() == 'true': return True
                 if val.lower() == 'false': return False
                 if val.lower() == 'none': return None
                 # Fallback to literal_eval only if it might be a container or complex literal
            return ast.literal_eval(val)
        except (ValueError, SyntaxError, TypeError, MemoryError) as e:
            eval_logger = logging.getLogger('utils.safe_eval')
            eval_logger.debug(f"safe_literal_eval failed for value '{val}': {e}. Returning as string.")
            return val # Return original string if eval fails
    return val # Return original value if not string

# --- MODIFIED HELPER: Returns Surplus AND Infra-marginal Count ---
def _calculate_potential_surplus_and_count(role, values_costs, eq_price):
    """
    Helper function to calculate theoretical individual surplus and
    the count of infra-marginal units at the equilibrium price.
    Returns: (potential_surplus, infra_marginal_count)
    """
    if eq_price is None or not values_costs:
        return 0.0, 0 # Cannot calculate without price or values

    potential_surplus = 0.0
    infra_marginal_count = 0
    try:
        # Ensure values/costs are numeric
        numeric_vals = [float(vc) for vc in values_costs if isinstance(vc, (int, float, np.number))]
        eq_p_float = float(eq_price)

        if role == 'buyer':
            for v in numeric_vals:
                if v > eq_p_float: # Strictly infra-marginal buyer unit (v > P)
                    potential_surplus += (v - eq_p_float)
                    infra_marginal_count += 1
                elif v == eq_p_float: # Marginal unit (v = P) - counts for surplus but maybe not unique count? Let's count it.
                    infra_marginal_count += 1 # Include marginal units in count

        elif role == 'seller':
            for c in numeric_vals:
                if c < eq_p_float: # Strictly infra-marginal seller unit (c < P)
                    potential_surplus += (eq_p_float - c)
                    infra_marginal_count += 1
                elif c == eq_p_float: # Marginal unit (c = P)
                    infra_marginal_count += 1 # Include marginal units in count

    except (ValueError, TypeError) as e:
        calc_logger = logging.getLogger('utils.calc_surplus')
        calc_logger.warning(f"Error calculating individual surplus/count for role={role}, vals={values_costs}, eq_p={eq_price}: {e}")
        return 0.0, 0 # Return 0s if error occurs

    return potential_surplus, infra_marginal_count

# --- MODIFIED Analysis: Calculates ProfitRatio and SuccessRate ---
def analyze_individual_performance(round_stats):
    """
    Aggregates performance by individual bot instance across rounds.
    Separates buyers and sellers and ranks them by Mean Profit within each role.
    Includes Mean Profit Capture Ratio and Mean Trading Success Rate.
    """
    ind_logger = logging.getLogger('analysis.individual')
    if not round_stats: return "No round stats for individual performance analysis."

    # Structure: { (role, strategy, name): {'profits': [], 'ratios': [], 'success_rates': [], 'trades': []}, ... }
    all_bots_data = defaultdict(lambda: {'profits': [], 'ratios': [], 'success_rates': [], 'trades': []})
    num_processed_rounds = 0

    # --- Stage 1: Collect profits and calculate ratios/rates ---
    for r_idx, rstat in enumerate(round_stats):
        if not isinstance(rstat, dict): continue
        bot_details_raw = rstat.get("bot_details")
        eq_price = rstat.get("eq_p") # Equilibrium price for the round

        if bot_details_raw is None or eq_price is None:
             ind_logger.debug(f"R{r_idx}: Skipping round analysis due to missing bot_details or eq_p.")
             continue

        # Use safe_literal_eval to parse the string back to list of dicts
        bot_details = safe_literal_eval(bot_details_raw)
        if not isinstance(bot_details, list) or not bot_details:
            ind_logger.warning(f"R{r_idx}: Could not parse bot_details: {bot_details_raw}")
            continue

        num_processed_rounds += 1
        for b_idx, b in enumerate(bot_details):
            if not isinstance(b, dict): continue

            role = b.get("role")
            strategy = b.get("strategy")
            name = b.get("name")
            profit_raw = b.get("profit")
            values_costs_raw = b.get("values_costs") # Original values/costs (might be string)
            trades_raw = b.get("trades")         # Number of trades made

            values_costs = safe_literal_eval(values_costs_raw) # Parse values/costs

            if None in [role, strategy, name, values_costs, trades_raw]:
                ind_logger.debug(f"R{r_idx} Bot {b_idx}: Skipping due to missing key info: {b}")
                continue
            key = (role, strategy, name)

            try:
                profit_val = float(profit_raw if profit_raw is not None else 0.0)
                num_trades = int(trades_raw)
                all_bots_data[key]['profits'].append(profit_val)
                all_bots_data[key]['trades'].append(num_trades)

                # Calculate potential surplus and infra-marginal count for the single period potential
                potential_surplus, infra_marginal_count = _calculate_potential_surplus_and_count(role, values_costs, eq_price)

                # Calculate Profit Capture Ratio (Profit / Potential Surplus)
                profit_ratio = 0.0 # Default
                if abs(potential_surplus) > 1e-9:
                    profit_ratio = profit_val / potential_surplus
                elif abs(profit_val) < 1e-9: # If surplus is ~0 and profit is ~0
                    profit_ratio = 1.0 # Define 0/0 as 1.0
                profit_ratio_clamped = np.clip(profit_ratio, -0.1, 2.0) # Allow ratio > 1, but clamp extremes
                all_bots_data[key]['ratios'].append(profit_ratio_clamped)

                # Calculate Trading Success Rate (Trades / Infra-marginal Count)
                success_rate = 0.0 # Default
                if infra_marginal_count > 0:
                    success_rate = num_trades / infra_marginal_count
                elif num_trades == 0: # If no infra-marginal units and no trades made
                    success_rate = 1.0 # Define 0/0 as 100% successful (traded all possible)
                success_rate_clamped = np.clip(success_rate, 0.0, 1.0) # Strictly between 0 and 1
                all_bots_data[key]['success_rates'].append(success_rate_clamped)


            except (ValueError, TypeError, KeyError) as e:
                ind_logger.warning(f"R{r_idx}: Error processing data for bot {key}: {e}. Vals={values_costs_raw}, Pft={profit_raw}, Trd={trades_raw}")

    if not all_bots_data:
        return f"No valid bot details found in {num_processed_rounds} processed rounds."

    # --- Stage 2: Calculate aggregate stats and separate by role ---
    buyer_results = []
    seller_results = []

    for key, data_dict in all_bots_data.items():
        role, strategy, name = key
        profits = np.array(data_dict['profits'])
        ratios = np.array(data_dict['ratios'])
        success_rates = np.array(data_dict['success_rates'])
        if len(profits) == 0: continue # Skip if no valid entries

        avg_profit = np.nanmean(profits); std_profit = np.nanstd(profits)
        min_p = np.nanmin(profits); med = np.nanmedian(profits); max_p = np.nanmax(profits)
        avg_ratio = np.nanmean(ratios); std_ratio = np.nanstd(ratios)
        avg_success = np.nanmean(success_rates); std_success = np.nanstd(success_rates)

        result_dict = {
            'role': role, 'strategy': strategy, 'name': name,
            'mean_profit': avg_profit, 'std_profit': std_profit,
            'min_profit': min_p, 'median_profit': med, 'max_profit': max_p,
            'mean_profit_ratio': avg_ratio, 'std_profit_ratio': std_ratio, # Renamed metric
            'mean_success_rate': avg_success, 'std_success_rate': std_success # New metric
        }

        if role == 'buyer': buyer_results.append(result_dict)
        elif role == 'seller': seller_results.append(result_dict)

    # --- Stage 3: Sort results by Mean Profit (descending) ---
    buyers_sorted = sorted(buyer_results, key=lambda x: x['mean_profit'], reverse=True)
    sellers_sorted = sorted(seller_results, key=lambda x: x['mean_profit'], reverse=True)

    # --- Stage 4: Format tables ---
    headers = ["Rank", "Strategy", "BotName", "MeanProfit", "StdProfit",
               "MeanProfitRatio", "StdProfitRatio", "MeanSuccessRate", "StdSuccessRate",
               "MinProfit", "MedianProfit", "MaxProfit"]
    buyer_table_rows = []
    seller_table_rows = []

    for rank, res in enumerate(buyers_sorted, 1):
        buyer_table_rows.append([
            rank, res['strategy'], res['name'],
            f"{res['mean_profit']:.2f}", f"{res['std_profit']:.2f}",
            f"{res['mean_profit_ratio']:.3f}", f"({res['std_profit_ratio']:.3f})", # Profit Ratio
            f"{res['mean_success_rate']:.3f}", f"({res['std_success_rate']:.3f})", # Success Rate
            f"{res['min_profit']:.2f}", f"{res['median_profit']:.2f}", f"{res['max_profit']:.2f}"
        ])

    for rank, res in enumerate(sellers_sorted, 1):
        seller_table_rows.append([
            rank, res['strategy'], res['name'],
            f"{res['mean_profit']:.2f}", f"{res['std_profit']:.2f}",
            f"{res['mean_profit_ratio']:.3f}", f"({res['std_profit_ratio']:.3f})", # Profit Ratio
            f"{res['mean_success_rate']:.3f}", f"({res['std_success_rate']:.3f})", # Success Rate
            f"{res['min_profit']:.2f}", f"{res['median_profit']:.2f}", f"{res['max_profit']:.2f}"
        ])

    title = f"\n=== INDIVIDUAL BOT PERFORMANCE (Aggregated over {num_processed_rounds} Rounds) ==="
    buyer_title = "\n--- Buyers (Ranked by Mean Profit) ---"
    seller_title = "\n--- Sellers (Ranked by Mean Profit) ---"

    buyer_table = tabulate(buyer_table_rows, headers=headers, tablefmt='pretty', floatfmt='.3f', numalign="right") if buyer_table_rows else "No buyer data."
    seller_table = tabulate(seller_table_rows, headers=headers, tablefmt='pretty', floatfmt='.3f', numalign="right") if seller_table_rows else "No seller data."

    final_output = f"{title}\n{buyer_title}\n{buyer_table}\n{seller_title}\n{seller_table}"
    return final_output


# --- MODIFIED Market Performance Analysis ---
def analyze_market_performance(round_stats):
    """ Aggregates market performance including standard deviations. """
    mkt_logger = logging.getLogger('analysis.market')
    if not round_stats: return "No round stats for market performance analysis."

    # Lists to store per-round metrics
    effs_raw, effs_clamped = [], []
    # Store deviations as relative % where possible, absolute otherwise
    price_devs_percent, quant_devs_percent = [], []
    price_devs_abs, quant_devs_abs = [], []

    buyer_surplus_fracs, seller_surplus_fracs = [], []
    avg_buyer_profits_per_round, avg_seller_profits_per_round = [], []

    eff_issues, neg_eff_rounds, num_processed_rounds = 0, 0, 0

    for r_idx, rstat in enumerate(round_stats):
        if not isinstance(rstat, dict): continue
        num_processed_rounds += 1

        # --- Efficiency ---
        eff_raw = rstat.get('market_efficiency')
        eff = np.nan
        try:
             if eff_raw is None: raise ValueError("Missing efficiency")
             eff = float(eff_raw)
             effs_raw.append(eff)
             is_issue = False
             if eff > 1.0 + 1e-6: is_issue = True # Check if significantly > 1
             if eff < -1e-9: neg_eff_rounds +=1; is_issue = True
             if is_issue: eff_issues += 1
             eff_clamped_val = np.clip(eff, 0.0, 1.0) # Hard clamp for averaging
             effs_clamped.append(eff_clamped_val)
        except (ValueError, TypeError, AttributeError) as e:
             mkt_logger.warning(f"R{r_idx}: Invalid or missing efficiency value '{eff_raw}': {e}. Using NaN.")
             effs_raw.append(np.nan)
             effs_clamped.append(np.nan)

        # --- Price/Quantity Deviation ---
        adp = rstat.get('abs_diff_price'); adq = rstat.get('abs_diff_quantity')
        eq_p = rstat.get('eq_p'); eq_q = rstat.get('eq_q') # Get equilibrium values

        # Price Deviation
        try:
             p_abs = float(adp) if adp is not None else np.nan
             price_devs_abs.append(p_abs)
             if not np.isnan(p_abs) and eq_p is not None and abs(eq_p) > 1e-6:
                 price_devs_percent.append(p_abs / abs(eq_p) * 100.0)
             else: price_devs_percent.append(np.nan)
        except (ValueError, TypeError): price_devs_abs.append(np.nan); price_devs_percent.append(np.nan)

        # Quantity Deviation
        try:
             q_abs = float(adq) if adq is not None else np.nan
             quant_devs_abs.append(q_abs)
             if not np.isnan(q_abs) and eq_q is not None and eq_q > 0:
                 quant_devs_percent.append(q_abs / eq_q * 100.0)
             else: quant_devs_percent.append(np.nan)
        except (ValueError, TypeError): quant_devs_abs.append(np.nan); quant_devs_percent.append(np.nan)


        # --- Surplus Split & Avg Profits ---
        role_perf_raw = rstat.get("role_strat_perf")
        round_total_b_profit, round_b_count = 0.0, 0
        round_total_s_profit, round_s_count = 0.0, 0
        valid_perf_data = False

        role_perf = safe_literal_eval(role_perf_raw) # Use safe eval
        if isinstance(role_perf, dict):
             valid_perf_data = True
             try:
                 for key_repr, perf_data in role_perf.items():
                     key_tuple = safe_literal_eval(key_repr) # Use safe eval
                     if not (isinstance(key_tuple, tuple) and len(key_tuple)==2 and isinstance(perf_data, dict)):
                          valid_perf_data = False; break
                     role, _ = key_tuple
                     profit = float(perf_data.get("profit", 0.0))
                     count = int(perf_data.get("count", 0))
                     if count < 0: raise ValueError("Negative count")
                     if role == 'buyer': round_total_b_profit += profit; round_b_count += count
                     elif role == 'seller': round_total_s_profit += profit; round_s_count += count
             except (ValueError, TypeError, KeyError) as e:
                  valid_perf_data = False
                  if num_processed_rounds == 1 or r_idx % 100 == 0: # Log periodically
                       mkt_logger.warning(f"R{r_idx}: Error processing role_strat_perf data '{role_perf_raw}': {e}")

        if valid_perf_data and (round_b_count + round_s_count > 0):
             tot_pft_roles = round_total_b_profit + round_total_s_profit
             buyer_frac = round_total_b_profit / tot_pft_roles if abs(tot_pft_roles) > 1e-9 else 0.5
             seller_frac = round_total_s_profit / tot_pft_roles if abs(tot_pft_roles) > 1e-9 else 0.5
             buyer_surplus_fracs.append(buyer_frac)
             seller_surplus_fracs.append(seller_frac)
             avg_buyer_profits_per_round.append(round_total_b_profit / round_b_count if round_b_count > 0 else 0.0)
             avg_seller_profits_per_round.append(round_total_s_profit / round_s_count if round_s_count > 0 else 0.0)
        else:
             buyer_surplus_fracs.append(np.nan); seller_surplus_fracs.append(np.nan)
             avg_buyer_profits_per_round.append(np.nan); avg_seller_profits_per_round.append(np.nan)
             # if num_processed_rounds == 1 or r_idx % 100 == 0:
             #      mkt_logger.warning(f"R{r_idx}: Could not calculate surplus split/avg profits due to missing/invalid role_strat_perf data: {role_perf_raw}")

    # --- Aggregate Results ---
    if num_processed_rounds == 0: return "Market analysis failed: No rounds processed."
    if eff_issues > 0:
         mkt_logger.warning(f"Market Analysis: {eff_issues}/{num_processed_rounds} rounds had raw efficiency outside [0, 1] (Negative: {neg_eff_rounds}). Clamped Efficiency used for avg/std calculation.")

    avg_eff, std_eff = (np.nanmean(effs_clamped), np.nanstd(effs_clamped)) if effs_clamped else (np.nan, np.nan)
    # Use absolute deviations for reporting as per paper table
    avg_p_dev_abs, std_p_dev_abs = (np.nanmean(price_devs_abs), np.nanstd(price_devs_abs)) if price_devs_abs else (np.nan, np.nan)
    avg_q_dev_abs, std_q_dev_abs = (np.nanmean(quant_devs_abs), np.nanstd(quant_devs_abs)) if quant_devs_abs else (np.nan, np.nan)

    avg_b_surplus, std_b_surplus = (np.nanmean(buyer_surplus_fracs), np.nanstd(buyer_surplus_fracs)) if buyer_surplus_fracs else (np.nan, np.nan)
    avg_s_surplus, std_s_surplus = (np.nanmean(seller_surplus_fracs), np.nanstd(seller_surplus_fracs)) if seller_surplus_fracs else (np.nan, np.nan)
    avg_b_prof, std_b_prof = (np.nanmean(avg_buyer_profits_per_round), np.nanstd(avg_buyer_profits_per_round)) if avg_buyer_profits_per_round else (np.nan, np.nan)
    avg_s_prof, std_s_prof = (np.nanmean(avg_seller_profits_per_round), np.nanstd(avg_seller_profits_per_round)) if avg_seller_profits_per_round else (np.nan, np.nan)

    # --- Format Output Strings ---
    def format_mean_std(mean_val, std_val, precision, is_percent=False):
        if np.isnan(mean_val) or np.isnan(std_val): return "NaN (NaN)"
        unit = "%" if is_percent else ""
        scale = 100.0 if is_percent else 1.0
        return f"{mean_val * scale:.{precision}f}{unit} ({std_val * scale:.{precision}f})"

    eff_str = format_mean_std(avg_eff, std_eff, 2) # Use clamped efficiency avg/std
    b_prof_str = format_mean_std(avg_b_prof, std_b_prof, 0)
    s_prof_str = format_mean_std(avg_s_prof, std_s_prof, 0)
    b_surplus_str = format_mean_std(avg_b_surplus, std_b_surplus, 1, is_percent=True)
    s_surplus_str = format_mean_std(avg_s_surplus, std_s_surplus, 1, is_percent=True)
    p_dev_str = format_mean_std(avg_p_dev_abs, std_p_dev_abs, 1) # Absolute Price Dev
    q_dev_str = format_mean_std(avg_q_dev_abs, std_q_dev_abs, 1) # Absolute Quant Dev

    # --- Create Table ---
    headers = ["Market Eff", "AvgBuyerProfit", "AvgSellerProfit",
               "BuyerSurplus%", "SellerSurplus%", "AvgPriceDev", "AvgQuantDev"]
    rows = [[eff_str, b_prof_str, s_prof_str, b_surplus_str, s_surplus_str, p_dev_str, q_dev_str]]
    title = f"\n=== MARKET PERFORMANCE (Mean (StdDev) over {num_processed_rounds} Rounds) ==="
    table = tabulate(rows, headers=headers, tablefmt="pretty", numalign="right")
    if eff_issues > 0:
        table += f"\nNote: Market Eff calculated using clamped [0,1] values for {eff_issues} rounds."

    return f"{title}\n{table}"


# --- MODIFIED Tournament Analysis: Includes Profit Ratio and Success Rate ---
def analyze_strategy_tournament(round_stats):
    """
    Analyzes performance aggregated by STRATEGY, ranking them like a tournament.
    Calculates mean/std for profit, rank, profit ratio, and success rate.
    """
    tourn_logger = logging.getLogger('analysis.tournament')
    if not round_stats: return "No round stats for tournament analysis."

    # Structure: { strategy_name: {'profits': [], 'ranks': [], 'ratios': [], 'success_rates': [] } }
    strategy_performance = defaultdict(lambda: {'profits': [], 'ranks': [], 'ratios': [], 'success_rates': []})
    num_processed_rounds = 0
    total_agents_processed = 0

    # --- Stage 1: Collect data and rank within rounds ---
    for r_idx, rstat in enumerate(round_stats):
        if not isinstance(rstat, dict): continue
        bot_details_raw = rstat.get("bot_details")
        eq_price = rstat.get("eq_p") # Get equilibrium price

        if bot_details_raw is None or eq_price is None: continue

        # Use safe_literal_eval to parse the string back to list of dicts
        bot_details = safe_literal_eval(bot_details_raw)
        if not isinstance(bot_details, list) or not bot_details: continue

        num_processed_rounds += 1
        round_bots = [] # Store dicts for ranking
        valid_bots_in_round = True
        for b_idx, b in enumerate(bot_details):
            if not isinstance(b, dict): valid_bots_in_round = False; continue

            strategy = b.get("strategy")
            profit_raw = b.get("profit")
            name = b.get("name", f"Unknown_R{r_idx}_B{b_idx}")
            values_costs_raw = b.get("values_costs") # Original values/costs (might be string)
            role = b.get("role")
            trades_raw = b.get("trades") # Number of trades

            values_costs = safe_literal_eval(values_costs_raw) # Parse list

            if None in [strategy, role, values_costs, trades_raw]:
                 valid_bots_in_round = False; continue
            try:
                profit = float(profit_raw if profit_raw is not None else 0.0)
                num_trades = int(trades_raw)
                round_bots.append({
                    'strategy': strategy, 'profit': profit, 'name': name,
                    'values_costs': values_costs, 'role': role, 'num_trades': num_trades
                    })
                total_agents_processed += 1
            except (ValueError, TypeError): valid_bots_in_round = False; continue

        if not valid_bots_in_round or not round_bots: continue

        # --- Rank bots and calculate metrics within this round ---
        try:
            df_round = pd.DataFrame(round_bots)
            df_round['rank'] = df_round['profit'].rank(method='average', ascending=False)

            # Calculate potential surplus and infra-marginal count
            surplus_count = df_round.apply(
                lambda row: _calculate_potential_surplus_and_count(row['role'], row['values_costs'], eq_price), axis=1
            )
            df_round['potential_surplus'] = surplus_count.apply(lambda x: x[0])
            df_round['infra_marginal_count'] = surplus_count.apply(lambda x: x[1])

            # Calculate Profit Capture Ratio
            df_round['profit_ratio'] = np.where(
                np.abs(df_round['potential_surplus']) > 1e-9,
                df_round['profit'] / df_round['potential_surplus'],
                np.where(np.abs(df_round['profit']) < 1e-9, 1.0, 0.0) # Define 0/0 as 1.0
            )
            df_round['profit_ratio'] = np.clip(df_round['profit_ratio'], -0.1, 2.0) # Clamp ratio

            # Calculate Trading Success Rate
            df_round['success_rate'] = np.where(
                df_round['infra_marginal_count'] > 0,
                df_round['num_trades'] / df_round['infra_marginal_count'],
                np.where(df_round['num_trades'] == 0, 1.0, 0.0) # 0/0 -> 1.0, X/0 -> 0.0 if X>0
            )
            df_round['success_rate'] = np.clip(df_round['success_rate'], 0.0, 1.0) # Clamp rate

            # --- Aggregate results per strategy for this round ---
            for _, row in df_round.iterrows():
                strat = row['strategy']
                strategy_performance[strat]['profits'].append(row['profit'])
                strategy_performance[strat]['ranks'].append(row['rank'])
                strategy_performance[strat]['ratios'].append(row['profit_ratio']) # Store profit ratio
                strategy_performance[strat]['success_rates'].append(row['success_rate']) # Store success rate
        except Exception as e:
             tourn_logger.error(f"Round {r_idx}: Error during ranking or aggregation: {e}", exc_info=True)

    # --- Stage 2: Calculate overall stats ---
    if not strategy_performance:
        return f"No valid strategy performance data found across {num_processed_rounds} processed rounds (Total agents processed: {total_agents_processed})."

    results = []
    for strategy, data in strategy_performance.items():
        profits = np.array(data['profits']); ranks = np.array(data['ranks']);
        ratios = np.array(data['ratios']); success_rates = np.array(data['success_rates'])
        if len(profits) == 0: continue

        mean_profit = np.nanmean(profits); std_profit = np.nanstd(profits)
        mean_rank = np.nanmean(ranks); std_rank = np.nanstd(ranks)
        mean_ratio = np.nanmean(ratios); std_ratio = np.nanstd(ratios) # Profit Ratio stats
        mean_success = np.nanmean(success_rates); std_success = np.nanstd(success_rates) # Success Rate stats
        count = len(profits)

        results.append({
            'Strategy': strategy, 'MeanProfit': mean_profit, 'StdProfit': std_profit,
            'MeanRank': mean_rank, 'StdRank': std_rank,
            'MeanProfitRatio': mean_ratio, 'StdProfitRatio': std_ratio, # Renamed metric
            'MeanSuccessRate': mean_success, 'StdSuccessRate': std_success, # New metric
            'Count': count
        })

    if not results: return f"Could not compute final stats for any strategy across {num_processed_rounds} processed rounds."

    # --- Stage 3: Sort and Format Table ---
    results_sorted = sorted(results, key=lambda x: x['MeanRank']) # Sort by Mean Rank

    table_rows = []
    for res in results_sorted:
        table_rows.append([
            res['Strategy'],
            f"{res['MeanProfit']:.2f}", f"({res['StdProfit']:.2f})",
            f"{res['MeanRank']:.2f}", f"({res['StdRank']:.2f})",
            f"{res['MeanProfitRatio']:.3f}", f"({res['StdProfitRatio']:.3f})", # Profit Ratio
            f"{res['MeanSuccessRate']:.3f}", f"({res['StdSuccessRate']:.3f})", # Success Rate
            res['Count']
        ])

    # Update Headers
    headers = ["Strategy", "Mean Profit", "(Std Dev)", "Mean Rank", "(Std Dev)",
               "MeanProfitRatio", "(Std Dev)", "MeanSuccessRate", "(Std Dev)",
               "Agent-Rounds"]
    title = f"\n=== STRATEGY TOURNAMENT RANKING (Aggregated over {num_processed_rounds} Rounds) ==="
    table = tabulate(table_rows, headers=headers, tablefmt="pretty", stralign="left", numalign="right")
    return f"{title}\n{table}"


# --- Plotting Functions ---
def plot_per_round(round_stats_list, exp_path, dfLogs=None, generate_plots=True):
    """ Placeholder for per-round plotting. """
    plot_logger = logging.getLogger('plotting.round')
    if not plt_available: plot_logger.warning("Plotting disabled (matplotlib unavailable)."); return
    if not generate_plots: plot_logger.info("Skipping per-round plots (disabled by config)."); return
    if dfLogs is None or dfLogs.empty: plot_logger.warning("No step log data for per-round plots."); return
    plot_logger.debug("plot_per_round called (plotting logic currently omitted for brevity).")
    # Add detailed plotting logic here if needed
    pass

def plot_rl_behavior_eval(dfR, dfLogs, config, exp_path, num_rounds_to_plot=5):
    """ Placeholder for plotting RL agent behavior during evaluation. """
    plot_logger = logging.getLogger('plotting.rl_eval')
    if not plt_available: plot_logger.warning("Plotting disabled (matplotlib unavailable)."); return
    if dfR is None or dfR.empty or dfLogs is None or dfLogs.empty:
         plot_logger.warning("Missing data for RL behavior plots.")
         return
    plot_logger.debug("plot_rl_behavior_eval called (plotting logic currently omitted for brevity).")
    # Add detailed plotting logic here if needed
    pass

def plot_ppo_training_curves(rl_training_logs, exp_path):
    """ Plots PPO training metrics (loss, entropy, etc.) over rounds. """
    plot_logger = logging.getLogger('plotting.ppo_train')
    if not plt_available: plot_logger.warning("Plotting disabled (matplotlib unavailable)."); return
    if not rl_training_logs: plot_logger.info("No RL training logs to plot."); return

    try:
        df_train = pd.DataFrame(rl_training_logs)
        if df_train.empty: plot_logger.info("RL training log DataFrame is empty."); return

        required_cols = ['round', 'avg_policy_loss', 'avg_value_loss', 'avg_entropy']
        if not all(col in df_train.columns for col in required_cols):
             missing = [col for col in required_cols if col not in df_train.columns]
             plot_logger.warning(f"RL training logs missing required columns {missing}. Cannot plot curves.")
             return

        df_grouped = df_train.groupby('round').agg(
            policy_loss = ('avg_policy_loss', 'mean'),
            value_loss = ('avg_value_loss', 'mean'),
            entropy = ('avg_entropy', 'mean'),
            approx_kl = ('avg_approx_kl', lambda x: np.nanmean(x.astype(float))),
            clip_frac = ('avg_clip_frac', lambda x: np.nanmean(x.astype(float)))
        ).reset_index()

        num_plots = 3
        plot_cols = ['policy_loss', 'value_loss', 'entropy']
        plot_labels = ['Policy Loss', 'Value Loss', 'Policy Entropy']
        plot_colors = ['C0', 'orange', 'green']

        if 'approx_kl' in df_grouped.columns:
             num_plots += 1; plot_cols.append('approx_kl'); plot_labels.append('Approx KL'); plot_colors.append('red')
        if 'clip_frac' in df_grouped.columns:
             num_plots += 1; plot_cols.append('clip_frac'); plot_labels.append('Clip Fraction'); plot_colors.append('purple')

        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots), sharex=True)
        if num_plots == 1: axes = [axes] # Ensure axes is iterable even with 1 plot
        exp_name_title = os.path.basename(os.path.normpath(exp_path))
        fig.suptitle(f'PPO Training Curves ({exp_name_title})')

        for i in range(num_plots):
            col = plot_cols[i]
            label = plot_labels[i]
            color = plot_colors[i]
            if col in df_grouped.columns and not df_grouped[col].isnull().all():
                 axes[i].plot(df_grouped['round'], df_grouped[col], label=label, color=color, alpha=0.8, linewidth=1.5)
                 axes[i].set_ylabel(label)
                 axes[i].grid(True, linestyle='--', alpha=0.6)
                 axes[i].legend()
            else:
                 axes[i].text(0.5, 0.5, f'{label} data not available', horizontalalignment='center', verticalalignment='center', transform=axes[i].transAxes)
                 axes[i].set_ylabel(label)
                 axes[i].grid(True, linestyle='--', alpha=0.6) # Still add grid

        axes[-1].set_xlabel('Training Round')
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        save_path = os.path.join(exp_path, "ppo_training_curves.png")
        try:
            plt.savefig(save_path)
            plot_logger.info(f"Saved PPO training curves plot to {save_path}")
        except Exception as e:
            plot_logger.error(f"Error saving PPO training plot: {e}")
        plt.close(fig)

    except Exception as e:
        plot_logger.error(f"Error generating PPO training curves plot: {e}", exc_info=True)


def plot_game_summary(dfR, exp_path, dfLogs=None):
    """ Plots overall game summary metrics (e.g., efficiency over rounds). """
    plot_logger = logging.getLogger('plotting.summary')
    if not plt_available: plot_logger.warning("Plotting disabled (matplotlib unavailable)."); return
    if dfR is None or dfR.empty: plot_logger.warning("No round data (dfR) to plot game summary."); return

    try:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        exp_name_title = os.path.basename(os.path.normpath(exp_path))
        fig.suptitle(f'Market Summary ({exp_name_title})')

        # --- Plot Efficiency ---
        if 'market_efficiency' in dfR.columns:
             eff_plot = np.clip(pd.to_numeric(dfR['market_efficiency'], errors='coerce').fillna(0), 0, 1)
             axes[0].plot(dfR['round'], eff_plot * 100, label='Market Efficiency', alpha=0.7, linewidth=1)
             window_size = max(1, min(50, len(dfR)//10))
             if len(dfR) >= window_size:
                 try:
                     rolling_eff = pd.Series(eff_plot).rolling(window=window_size, center=True, min_periods=1).mean()
                     axes[0].plot(dfR['round'], rolling_eff * 100, label=f'{window_size}-Round Avg Eff', color='red', linestyle='--', linewidth=1.5)
                 except Exception as roll_e: plot_logger.warning(f"Could not calculate rolling efficiency: {roll_e}")
             axes[0].set_ylabel('Efficiency (%)'); axes[0].set_ylim(-5, 105); axes[0].grid(True, linestyle='--', alpha=0.6); axes[0].legend()
        else:
             axes[0].text(0.5, 0.5, 'Market Efficiency data not found', ha='center', va='center', transform=axes[0].transAxes); axes[0].set_ylabel('Efficiency (%)')

        # --- Plot Price Deviation ---
        # Use absolute deviation by default, relative if possible
        if 'abs_diff_price' in dfR.columns:
            abs_diff = pd.to_numeric(dfR['abs_diff_price'], errors='coerce')
            use_relative = False
            if 'eq_p' in dfR.columns:
                eq_p_safe = pd.to_numeric(dfR['eq_p'], errors='coerce').replace(0, np.nan)
                if not eq_p_safe.isnull().all():
                    rel_price_dev = (abs_diff / eq_p_safe.abs() * 100) # Use abs() for denominator
                    use_relative = True

            if use_relative:
                plot_data = rel_price_dev.fillna(0)
                ylabel = 'Avg Price Dev from Eq (%)'
            else:
                plot_data = abs_diff.fillna(0)
                ylabel = 'Avg Abs Price Dev from Eq'

            axes[1].plot(dfR['round'], plot_data, label=ylabel.replace('Avg ', ''), alpha=0.7, linewidth=1)
            window_size = max(1, min(50, len(dfR)//10))
            if len(dfR) >= window_size:
                 try:
                     rolling_dev = pd.Series(plot_data).rolling(window=window_size, center=True, min_periods=1).mean()
                     axes[1].plot(dfR['round'], rolling_dev, label=f'{window_size}-Round Avg Dev', color='red', linestyle='--', linewidth=1.5)
                 except Exception as roll_e: plot_logger.warning(f"Could not calculate rolling price deviation: {roll_e}")
            axes[1].set_ylabel(ylabel); axes[1].grid(True, linestyle='--', alpha=0.6); axes[1].legend(); axes[1].set_ylim(bottom=-1 if use_relative else 0)
        else:
             axes[1].text(0.5, 0.5, 'Price Deviation data not found', ha='center', va='center', transform=axes[1].transAxes); axes[1].set_ylabel('Avg Price Dev from Eq')

        axes[-1].set_xlabel('Round')
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        save_path = os.path.join(exp_path, "market_summary_plot.png")
        try:
            plt.savefig(save_path)
            plot_logger.info(f"Saved market summary plot to {save_path}")
        except Exception as e:
            plot_logger.error(f"Error saving market summary plot: {e}")
        plt.close(fig)

    except Exception as e:
        plot_logger.error(f"Error generating game summary plot: {e}", exc_info=True)


# --- Additional Analysis Functions for Three-Phase Roadmap ---

def calculate_strategy_rankings(round_stats, traders_data):
    """
    Calculate comprehensive strategy performance rankings.
    Returns a DataFrame with strategy-level statistics.
    """
    analysis_logger = logging.getLogger('utils.analysis')
    
    try:
        strategy_stats = defaultdict(lambda: {
            'total_profit': 0,
            'total_trades': 0,
            'num_agents': 0,
            'rounds_participated': 0,
            'avg_efficiency_contribution': []
        })
        
        for round_stat in round_stats:
            if 'bot_details' not in round_stat:
                continue
                
            round_efficiency = round_stat.get('market_efficiency', 0)
            
            for bot in round_stat['bot_details']:
                strategy = bot.get('strategy', 'unknown')
                strategy_stats[strategy]['num_agents'] += 1
                strategy_stats[strategy]['total_profit'] += bot.get('profit', 0)
                strategy_stats[strategy]['total_trades'] += bot.get('trades', 0)
                strategy_stats[strategy]['rounds_participated'] += 1
                
                # Estimate efficiency contribution
                if bot.get('trades', 0) > 0:
                    strategy_stats[strategy]['avg_efficiency_contribution'].append(round_efficiency)
        
        # Calculate averages and create DataFrame
        rankings = []
        for strategy, stats in strategy_stats.items():
            avg_profit = stats['total_profit'] / max(1, stats['rounds_participated'])
            avg_trades = stats['total_trades'] / max(1, stats['rounds_participated'])
            avg_efficiency = np.mean(stats['avg_efficiency_contribution']) if stats['avg_efficiency_contribution'] else 0
            
            rankings.append({
                'strategy': strategy,
                'avg_profit_per_round': avg_profit,
                'avg_trades_per_round': avg_trades,
                'total_profit': stats['total_profit'],
                'total_trades': stats['total_trades'],
                'rounds_participated': stats['rounds_participated'],
                'avg_efficiency_contribution': avg_efficiency,
            })
        
        df_rankings = pd.DataFrame(rankings)
        df_rankings = df_rankings.sort_values('avg_profit_per_round', ascending=False)
        
        analysis_logger.info(f"Strategy rankings calculated for {len(rankings)} strategies")
        return df_rankings
        
    except Exception as e:
        analysis_logger.error(f"Error calculating strategy rankings: {e}")
        return pd.DataFrame()


def generate_competition_matrix(pairwise_results):
    """
    Generate a competition matrix from pairwise experiment results.
    Returns a DataFrame showing performance of each strategy pair.
    """
    analysis_logger = logging.getLogger('utils.analysis')
    
    try:
        # Extract unique strategies
        strategies = set()
        results_dict = {}
        
        for result in pairwise_results:
            buyer_strat = result.get('buyer_strategy')
            seller_strat = result.get('seller_strategy')
            if buyer_strat and seller_strat:
                strategies.add(buyer_strat)
                strategies.add(seller_strat)
                key = (buyer_strat, seller_strat)
                results_dict[key] = result.get('market_efficiency', 0)
        
        strategies = sorted(list(strategies))
        
        # Create matrix
        matrix = pd.DataFrame(index=strategies, columns=strategies)
        
        for buyer in strategies:
            for seller in strategies:
                if (buyer, seller) in results_dict:
                    matrix.loc[buyer, seller] = results_dict[(buyer, seller)]
                else:
                    matrix.loc[buyer, seller] = np.nan
        
        analysis_logger.info(f"Competition matrix generated for {len(strategies)} strategies")
        return matrix
        
    except Exception as e:
        analysis_logger.error(f"Error generating competition matrix: {e}")
        return pd.DataFrame()


def analyze_ppo_performance(round_stats, identify_ppo_agents=lambda x: 'ppo' in x.lower()):
    """
    Analyze PPO agent performance against classical strategies.
    Returns detailed statistics about PPO performance.
    """
    analysis_logger = logging.getLogger('utils.analysis')
    
    try:
        ppo_stats = {
            'rounds_analyzed': 0,
            'ppo_profits': [],
            'classical_profits': [],
            'ppo_trades': [],
            'classical_trades': [],
            'ppo_win_rate': 0,
            'efficiency_with_ppo': [],
            'efficiency_without_ppo': []
        }
        
        ppo_wins = 0
        total_comparisons = 0
        
        for round_stat in round_stats:
            if 'bot_details' not in round_stat:
                continue
            
            ppo_stats['rounds_analyzed'] += 1
            round_ppo_profit = 0
            round_classical_profit = 0
            round_ppo_trades = 0
            round_classical_trades = 0
            has_ppo = False
            
            for bot in round_stat['bot_details']:
                strategy = bot.get('strategy', '')
                profit = bot.get('profit', 0)
                trades = bot.get('trades', 0)
                
                if identify_ppo_agents(strategy):
                    has_ppo = True
                    round_ppo_profit += profit
                    round_ppo_trades += trades
                else:
                    round_classical_profit += profit
                    round_classical_trades += trades
            
            if has_ppo:
                ppo_stats['ppo_profits'].append(round_ppo_profit)
                ppo_stats['ppo_trades'].append(round_ppo_trades)
                ppo_stats['classical_profits'].append(round_classical_profit)
                ppo_stats['classical_trades'].append(round_classical_trades)
                ppo_stats['efficiency_with_ppo'].append(round_stat.get('market_efficiency', 0))
                
                if round_ppo_profit > round_classical_profit:
                    ppo_wins += 1
                total_comparisons += 1
            else:
                ppo_stats['efficiency_without_ppo'].append(round_stat.get('market_efficiency', 0))
        
        # Calculate summary statistics
        if total_comparisons > 0:
            ppo_stats['ppo_win_rate'] = (ppo_wins / total_comparisons) * 100
        
        ppo_stats['avg_ppo_profit'] = np.mean(ppo_stats['ppo_profits']) if ppo_stats['ppo_profits'] else 0
        ppo_stats['avg_classical_profit'] = np.mean(ppo_stats['classical_profits']) if ppo_stats['classical_profits'] else 0
        ppo_stats['avg_ppo_trades'] = np.mean(ppo_stats['ppo_trades']) if ppo_stats['ppo_trades'] else 0
        ppo_stats['avg_classical_trades'] = np.mean(ppo_stats['classical_trades']) if ppo_stats['classical_trades'] else 0
        
        analysis_logger.info(f"PPO performance analyzed: win rate={ppo_stats['ppo_win_rate']:.1f}%")
        return ppo_stats
        
    except Exception as e:
        analysis_logger.error(f"Error analyzing PPO performance: {e}")
        return {}


def detect_emergent_behaviors(step_logs, round_stats):
    """
    Detect emergent behaviors in multi-agent RL scenarios.
    Looks for patterns like collusion, manipulation, and convergence.
    """
    analysis_logger = logging.getLogger('utils.analysis')
    
    try:
        behaviors = {
            'price_clustering': [],
            'bid_ask_spread_evolution': [],
            'trade_velocity_changes': [],
            'agent_specialization': {},
            'potential_collusion': False,
            'market_manipulation_events': []
        }
        
        if not step_logs:
            return behaviors
        
        df_steps = pd.DataFrame(step_logs)
        
        # Analyze price clustering
        if 'trade_price' in df_steps.columns:
            trades_df = df_steps[df_steps['trade_executed'] == 1]
            if len(trades_df) > 0:
                for round_num in trades_df['round'].unique():
                    round_trades = trades_df[trades_df['round'] == round_num]['trade_price']
                    if len(round_trades) > 5:
                        # Calculate coefficient of variation
                        cv = round_trades.std() / round_trades.mean() if round_trades.mean() != 0 else 0
                        behaviors['price_clustering'].append(cv)
        
        # Analyze bid-ask spread
        if 'best_bid' in df_steps.columns and 'best_ask' in df_steps.columns:
            df_steps['spread'] = df_steps['best_ask'] - df_steps['best_bid']
            spread_by_round = df_steps.groupby('round')['spread'].mean()
            behaviors['bid_ask_spread_evolution'] = spread_by_round.tolist()
            
            # Check for potential collusion (persistently wide spreads)
            if len(spread_by_round) > 10:
                recent_spreads = spread_by_round.tail(10)
                avg_spread = recent_spreads.mean()
                early_spread = spread_by_round.head(10).mean()
                if avg_spread > early_spread * 1.5:
                    behaviors['potential_collusion'] = True
        
        # Analyze trade velocity
        trades_per_round = df_steps.groupby('round')['trade_executed'].sum()
        behaviors['trade_velocity_changes'] = trades_per_round.tolist()
        
        # Detect manipulation events (sudden price spikes/drops)
        if 'trade_price' in df_steps.columns:
            trades_df = df_steps[df_steps['trade_executed'] == 1]
            if len(trades_df) > 0:
                price_changes = trades_df['trade_price'].pct_change()
                manipulation_threshold = 0.2  # 20% sudden change
                manipulation_events = price_changes[abs(price_changes) > manipulation_threshold]
                behaviors['market_manipulation_events'] = len(manipulation_events)
        
        analysis_logger.info(f"Emergent behavior analysis complete. Potential collusion: {behaviors['potential_collusion']}")
        return behaviors
        
    except Exception as e:
        analysis_logger.error(f"Error detecting emergent behaviors: {e}")
        return behaviors


def calculate_information_value(results_by_info_level):
    """
    Calculate the value of information by comparing performance across information levels.
    """
    analysis_logger = logging.getLogger('utils.analysis')
    
    try:
        info_value = {}
        
        base_performance = results_by_info_level.get('base', {})
        base_profit = base_performance.get('avg_profit', 0)
        base_efficiency = base_performance.get('avg_efficiency', 0)
        
        for level, performance in results_by_info_level.items():
            if level == 'base':
                info_value[level] = {
                    'profit_gain': 0,
                    'efficiency_gain': 0,
                    'absolute_profit': base_profit,
                    'absolute_efficiency': base_efficiency
                }
            else:
                profit = performance.get('avg_profit', 0)
                efficiency = performance.get('avg_efficiency', 0)
                
                info_value[level] = {
                    'profit_gain': profit - base_profit,
                    'efficiency_gain': efficiency - base_efficiency,
                    'absolute_profit': profit,
                    'absolute_efficiency': efficiency,
                    'profit_gain_pct': ((profit - base_profit) / base_profit * 100) if base_profit != 0 else 0,
                    'efficiency_gain_pct': ((efficiency - base_efficiency) / base_efficiency * 100) if base_efficiency != 0 else 0
                }
        
        analysis_logger.info(f"Information value calculated for {len(info_value)} levels")
        return info_value
        
    except Exception as e:
        analysis_logger.error(f"Error calculating information value: {e}")
        return {}


def generate_latex_table(df, caption, label, filename=None):
    """
    Generate a LaTeX table from a DataFrame.
    """
    analysis_logger = logging.getLogger('utils.analysis')
    
    try:
        # Convert DataFrame to LaTeX
        latex_str = df.to_latex(index=False, 
                                float_format="%.3f",
                                caption=caption,
                                label=label,
                                column_format='l' + 'r' * (len(df.columns) - 1))
        
        # Add booktabs formatting
        latex_str = latex_str.replace('\\toprule', '\\toprule\n\\midrule')
        latex_str = latex_str.replace('\\bottomrule', '\\midrule\n\\bottomrule')
        
        if filename:
            with open(filename, 'w') as f:
                f.write(latex_str)
            analysis_logger.info(f"LaTeX table saved to {filename}")
        
        return latex_str
        
    except Exception as e:
        analysis_logger.error(f"Error generating LaTeX table: {e}")
        return ""


def create_publication_plots(round_stats, step_logs, output_dir):
    """
    Create publication-quality plots for the research paper.
    """
    plot_logger = logging.getLogger('utils.plotting')
    
    if not plt_available:
        plot_logger.warning("Matplotlib not available, skipping publication plots")
        return
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Set publication style
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['axes.titlesize'] = 11
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
        plt.rcParams['figure.dpi'] = 300
        
        # Create multi-panel figure
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # Panel 1: Efficiency over time
        df_rounds = pd.DataFrame(round_stats)
        if 'market_efficiency' in df_rounds.columns:
            axes[0, 0].plot(df_rounds.index, df_rounds['market_efficiency'] * 100, 
                          alpha=0.5, color='blue', linewidth=0.5)
            
            # Add rolling average
            window = min(50, len(df_rounds) // 10)
            if window > 1:
                rolling_eff = df_rounds['market_efficiency'].rolling(window=window).mean()
                axes[0, 0].plot(df_rounds.index, rolling_eff * 100, 
                              color='red', linewidth=2, label=f'{window}-round MA')
            
            axes[0, 0].set_xlabel('Round')
            axes[0, 0].set_ylabel('Market Efficiency (%)')
            axes[0, 0].set_title('Market Efficiency Evolution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Panel 2: Trade volume
        if 'actual_trades' in df_rounds.columns:
            axes[0, 1].bar(df_rounds.index, df_rounds['actual_trades'], 
                         alpha=0.7, color='green', width=1)
            axes[0, 1].set_xlabel('Round')
            axes[0, 1].set_ylabel('Number of Trades')
            axes[0, 1].set_title('Trading Activity')
            axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Panel 3: Price convergence
        df_steps = pd.DataFrame(step_logs)
        if 'trade_price' in df_steps.columns and 'round' in df_steps.columns:
            trades_df = df_steps[df_steps['trade_executed'] == 1]
            
            # Sample rounds for clarity
            num_rounds_to_plot = min(5, trades_df['round'].nunique())
            rounds_to_plot = trades_df['round'].unique()[:num_rounds_to_plot]
            
            colors = plt.cm.viridis(np.linspace(0, 1, num_rounds_to_plot))
            
            for i, round_num in enumerate(rounds_to_plot):
                round_trades = trades_df[trades_df['round'] == round_num]
                axes[1, 0].scatter(round_trades['step'], round_trades['trade_price'],
                                 alpha=0.6, s=20, color=colors[i], label=f'Round {round_num}')
            
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Trade Price')
            axes[1, 0].set_title('Price Discovery Dynamics')
            axes[1, 0].legend(loc='best', framealpha=0.5)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Panel 4: Strategy performance comparison
        if 'bot_details' in round_stats[-1]:
            strategy_profits = defaultdict(list)
            for bot in round_stats[-1]['bot_details']:
                strategy = bot.get('strategy', 'unknown')
                profit = bot.get('profit', 0)
                strategy_profits[strategy].append(profit)
            
            strategies = list(strategy_profits.keys())
            avg_profits = [np.mean(strategy_profits[s]) for s in strategies]
            
            axes[1, 1].barh(strategies, avg_profits, color='orange', alpha=0.7)
            axes[1, 1].set_xlabel('Average Profit')
            axes[1, 1].set_ylabel('Strategy')
            axes[1, 1].set_title('Strategy Performance')
            axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Santa Fe Double Auction: Experimental Results', fontsize=12, y=1.02)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'publication_figure.pdf')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plot_logger.info(f"Publication figure saved to {output_path}")
        
        plt.close(fig)
        
    except Exception as e:
        plot_logger.error(f"Error creating publication plots: {e}")


def calculate_convergence_metrics(round_stats, window=50):
    """
    Calculate convergence metrics for Nash equilibrium analysis.
    """
    analysis_logger = logging.getLogger('utils.analysis')
    
    try:
        metrics = {
            'efficiency_variance': [],
            'price_variance': [],
            'converged': False,
            'convergence_round': None
        }
        
        df = pd.DataFrame(round_stats)
        
        # Calculate rolling variance
        if 'market_efficiency' in df.columns:
            eff_variance = df['market_efficiency'].rolling(window=window).var()
            metrics['efficiency_variance'] = eff_variance.tolist()
            
            # Check for convergence (variance below threshold)
            recent_variance = eff_variance.tail(window).mean()
            if recent_variance < 0.001:  # Threshold for convergence
                metrics['converged'] = True
                # Find convergence point
                for i in range(len(eff_variance) - window):
                    if all(eff_variance.iloc[i:i+window] < 0.001):
                        metrics['convergence_round'] = i
                        break
        
        analysis_logger.info(f"Convergence analysis complete. Converged: {metrics['converged']}")
        return metrics
        
    except Exception as e:
        analysis_logger.error(f"Error calculating convergence metrics: {e}")
        return metrics