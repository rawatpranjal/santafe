# main.py
import os
import sys
import shutil
import pprint
import pandas as pd
import logging
import argparse
import numpy as np
import importlib.util # Needed for dynamic config loading
import time # For basic timing if needed

# --- Argument Parsing ---
# Define expected arguments, many will override config values
parser = argparse.ArgumentParser(description="Run SFI Double Auction Simulation")
parser.add_argument('--config_file', type=str, required=True, help='Path to the Python configuration file to use (e.g., config_01a.py)')
parser.add_argument('--log', type=str, default=None, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Override console logging level from config')
parser.add_argument('--log_rl', type=str, default=None, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Override RL logic logging level from config')
parser.add_argument('--name', type=str, default=None, help='Override experiment name from config')
parser.add_argument('--num_rounds', type=int, default=None, help='Override total number of rounds from config')
parser.add_argument('--train_rounds', type=int, default=None, help='Override number of training rounds from config')
parser.add_argument('--load_model', type=str, default="<NOT_SET>", help='Override path *prefix* to load model. Use "None" string to disable loading if config has a path.')
parser.add_argument('--no_save', action='store_true', default=None, help="Disable saving RL model (overrides config if set)")
parser.add_argument('--no_round_plots', action='store_true', default=None, help="Disable per-round plots (overrides config if set)")
parser.add_argument('--no_eval_plots', action='store_true', default=None, help="Disable RL agent eval behavior plots (overrides config if set)")
args = parser.parse_args()

# --- Dynamically Load Configuration ---
config_path = args.config_file
if not os.path.isfile(config_path): # Check if it's a file
    print(f"ERROR: Configuration file not found or is not a file: {config_path}", file=sys.stderr)
    sys.exit(1)

try:
    # Create a module spec from the file path
    module_name = os.path.splitext(os.path.basename(config_path))[0] # Use filename as module name
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    if spec is None or spec.loader is None:
         raise ImportError(f"Could not create module spec for {config_path}")

    # Create a new module based on the spec
    config_module = importlib.util.module_from_spec(spec)
    # Execute the module (this runs the code in the config file)
    spec.loader.exec_module(config_module)

    # Get the CONFIG dictionary from the loaded module
    if not hasattr(config_module, 'CONFIG') or not isinstance(config_module.CONFIG, dict):
        raise AttributeError(f"'CONFIG' dictionary not found or is not a dict in {config_path}")
    CONFIG = config_module.CONFIG
    print(f"--- Successfully loaded config from: {config_path} ---")

except Exception as e:
    print(f"ERROR: Failed to load or process configuration from {config_path}: {e}", file=sys.stderr)
    sys.exit(1)

# --- Override config with CLI args IF they were provided ---
# Check if args have non-default values before overriding
if args.log is not None: CONFIG['log_level'] = args.log
if args.log_rl is not None: CONFIG['log_level_rl'] = args.log_rl
if args.name is not None: CONFIG['experiment_name'] = args.name
if args.num_rounds is not None: CONFIG['num_rounds'] = args.num_rounds
if args.train_rounds is not None: CONFIG['num_training_rounds'] = args.train_rounds
# Handle load_model override carefully: Allow setting to None via CLI
if args.load_model != "<NOT_SET>": # Check if argument was passed
    CONFIG['load_rl_model_path'] = None if args.load_model.lower() == "none" else args.load_model
# Action flags override only if specified (args.X will be True if flag is present, None if not)
if args.no_save is True: CONFIG['save_rl_model'] = False
if args.no_round_plots is True: CONFIG['generate_per_round_plots'] = False
if args.no_eval_plots is True: CONFIG['generate_eval_behavior_plots'] = False
# --- End Config Override ---


# --- Now use the final CONFIG dictionary ---
# Import local modules AFTER loading config
try:
    from auction import Auction
    from utils import (
        analyze_individual_performance,
        analyze_market_performance,
        analyze_strategy_tournament, # <-- IMPORT NEW FUNCTION
        plot_per_round,
        plot_game_summary,
        plot_rl_behavior_eval,
        plot_ppo_training_curves
    )
    from traders.registry import available_strategies
except ImportError as e:
    print(f"ERROR: Failed to import necessary local modules (Auction, utils, traders): {e}", file=sys.stderr)
    print("Ensure these files exist and are in the correct location relative to main.py.")
    sys.exit(1)


# --- Logging Setup ---
log_level_default_str = CONFIG.get('log_level', 'INFO').upper()
log_level_rl_str = CONFIG.get('log_level_rl', 'WARNING').upper()
# Ensure levels are valid logging levels
try:
    log_level_default = getattr(logging, log_level_default_str)
    log_level_rl = getattr(logging, log_level_rl_str)
except AttributeError:
    print(f"Warning: Invalid log level string found ('{log_level_default_str}' or '{log_level_rl_str}'). Defaulting to INFO/WARNING.")
    log_level_default = logging.INFO
    log_level_rl = logging.WARNING


log_format_file = '%(asctime)s - %(levelname)-8s - %(name)-15s - %(message)s'; date_format = '%Y-%m-%d %H:%M:%S'
log_format_console = '%(levelname)-8s - %(name)-15s - %(message)s'

# Basic configuration (will be overridden if file logging works)
logging.basicConfig(level=log_level_default, format=log_format_console, stream=sys.stdout, force=True)

# Prepare paths
exp_base_dir = CONFIG.get("experiment_dir", "experiments")
exp_name = CONFIG.get("experiment_name", "default_exp") # Get final name after override
exp_path = os.path.join(exp_base_dir, exp_name)
os.makedirs(exp_path, exist_ok=True)
log_file_path = os.path.join(exp_path, f"{exp_name}_run.log") # Include exp name in log file

# Attempt File Logging Setup
if CONFIG.get('log_to_file', True):
    try:
        root_logger = logging.getLogger()
        # Clear existing handlers to avoid duplicates if re-running in interactive session
        for h in root_logger.handlers[:]:
             root_logger.removeHandler(h)
             h.close() # Close file handle if open

        # File Handler (logs DEBUG and up by default)
        file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
        file_formatter = logging.Formatter(log_format_file, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG) # Log everything to file

        # Console Handler (logs level specified by log_level_default)
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(log_format_console)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(log_level_default) # Use configured level for console

        # Add handlers and set root level to lowest handler level (DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        root_logger.setLevel(logging.DEBUG) # Root logger captures everything >= DEBUG

        # Explicitly set levels for specific loggers AFTER root is configured
        # This allows finer control (e.g., keep auction logs INFO but trader DEBUG)
        logging.getLogger('auction').setLevel(log_level_default) # Auction log level
        logging.getLogger('trader').setLevel(log_level_default) # Basic trader logs (non-RL)
        logging.getLogger('utils').setLevel(log_level_default) # Util logs
        logging.getLogger('analysis').setLevel(log_level_default) # Analysis logs
        logging.getLogger('plotting').setLevel(log_level_default) # Plotting logs

        # Set levels for specific trader types if needed (e.g., rl_agent_logic logger)
        # Match logger names used in your RL core files
        rl_logic_logger_names = [
            'rl_agent_logic.ppo_lstm',
            'rl_agent_logic.ppo', # Add others if you have more
            'trader' # Also set base trader logger if RL agents use it
        ]
        # Apply the specific RL log level
        for name in rl_logic_logger_names:
             # Ensure the logger exists before setting level? No, logging.getLogger creates it.
             logging.getLogger(name).setLevel(log_level_rl)


        print(f"Logging detailed output to: {log_file_path}")
        print(f"Console log level set to: {log_level_default_str}")
        print(f"RL logic log level set to: {log_level_rl_str}")


    except Exception as e:
        print(f"Warning: File logging setup failed: {e}. Falling back to console logging only.")
        # Fallback basicConfig if file logging failed
        logging.basicConfig(level=log_level_default, format=log_format_console, stream=sys.stdout, force=True)
        # Re-apply specific levels in fallback
        logging.getLogger('auction').setLevel(log_level_default)
        logging.getLogger('trader').setLevel(log_level_default)
        logging.getLogger('utils').setLevel(log_level_default)
        logging.getLogger('analysis').setLevel(log_level_default)
        logging.getLogger('plotting').setLevel(log_level_default)
        rl_logic_logger_names = ['rl_agent_logic.ppo_lstm', 'rl_agent_logic.ppo', 'trader']
        for name in rl_logic_logger_names:
             logging.getLogger(name).setLevel(log_level_rl)
else:
    # File logging disabled, ensure specific levels are still set on console output
    logging.getLogger('auction').setLevel(log_level_default)
    logging.getLogger('trader').setLevel(log_level_default)
    logging.getLogger('utils').setLevel(log_level_default)
    logging.getLogger('analysis').setLevel(log_level_default)
    logging.getLogger('plotting').setLevel(log_level_default)
    rl_logic_logger_names = ['rl_agent_logic.ppo_lstm', 'rl_agent_logic.ppo', 'trader']
    for name in rl_logic_logger_names:
         logging.getLogger(name).setLevel(log_level_rl)
    print(f"File logging disabled. Console log level set to: {log_level_default_str}")
    print(f"RL logic log level set to: {log_level_rl_str}")


# Get logger for the main script itself
logger = logging.getLogger('main')

def main():
    """ Main function to run the auction simulation. """
    global CONFIG # Ensure main uses the globally loaded/modified CONFIG

    start_time = time.time()
    logger.info(f"Starting experiment: {CONFIG['experiment_name']}")
    logger.info(f"Results directory: {exp_path}")
    logger.info(f"Using parameters from: {args.config_file}")
    logger.info(f"Available strategies: {available_strategies()}")

    # Copy the *actually used* config file for reproducibility
    try:
        # Use the original config path provided as argument
        shutil.copy(args.config_file, os.path.join(exp_path, "config_used.py"))
    except Exception as e:
        logger.warning(f"Could not copy config file '{args.config_file}': {e}")

    # Log the final configuration being used (after potential overrides)
    logger.info("=== FINAL CONFIGURATION ===\n" + pprint.pformat(CONFIG, indent=2, width=100) + "\n=========================")

    # --- Run Auction ---
    auction_instance = None
    run_successful = False
    try:
        auction_instance = Auction(CONFIG) # Pass the final config dict
        auction_instance.run_auction()
        logger.info("Auction run completed successfully.")
        run_successful = True
    except KeyboardInterrupt:
         logger.warning("Auction run interrupted by user (KeyboardInterrupt).")
         # Optionally save partial results if auction_instance exists
         # Consider setting run_successful = False or partial flag
    except Exception as e:
        logger.exception("FATAL ERROR during auction run!")
        # Keep run_successful = False

    # --- Save Results (Even if run failed, try to save what exists) ---
    all_round_stats = auction_instance.round_stats if auction_instance else []
    all_step_logs = auction_instance.all_step_logs if auction_instance else []
    rl_training_logs = auction_instance.rl_training_logs if auction_instance else []
    dfR_all, dfLogs_all = None, None

    if all_round_stats:
        try:
            dfR_all = pd.DataFrame(all_round_stats)
            # Handle complex columns before saving to CSV
            complex_cols = ['buyer_vals', 'seller_vals', 'role_strat_perf', 'bot_details']
            for col in complex_cols:
                 if col in dfR_all.columns:
                     # Convert objects safely to strings for CSV compatibility
                     try:
                          dfR_all[col] = dfR_all[col].apply(lambda x: str(x) if x is not None else '')
                     except Exception as ser_e:
                          logger.warning(f"Could not serialize column {col} fully for CSV: {ser_e}. Saving as best effort.")
            round_log_path = os.path.join(exp_path, "round_log_all.csv")
            dfR_all.to_csv(round_log_path, index=False)
            logger.info(f"Saved {len(dfR_all)} round stats to {round_log_path}")
        except Exception as e:
            logger.error(f"Error saving round_log_all.csv: {e}", exc_info=True)
            dfR_all = None # Ensure it's None if saving failed
    else:
        logger.warning("No round stats generated or available to save.")

    if all_step_logs:
        try:
            dfLogs_all = pd.DataFrame(all_step_logs)
            # Handle complex columns before saving step log
            complex_cols_step = ['bids_submitted', 'asks_submitted']
            for col in complex_cols_step:
                 if col in dfLogs_all.columns:
                      try:
                          dfLogs_all[col] = dfLogs_all[col].apply(lambda x: str(x) if x is not None else '')
                      except Exception as ser_e:
                          logger.warning(f"Could not serialize column {col} fully for step log CSV: {ser_e}. Saving as best effort.")
            log_csv_path = os.path.join(exp_path, "step_log_all.csv")
            dfLogs_all.to_csv(log_csv_path, index=False)
            logger.info(f"Saved {len(dfLogs_all)} step logs to {log_csv_path}")
        except Exception as e:
            logger.error(f"Error saving step_log_all.csv: {e}", exc_info=True)
            dfLogs_all = None
    else:
        logger.warning("No step logs generated or available to save.")

    # --- Analysis & Plotting (Only if run seemed successful and data exists) ---
    if run_successful and dfR_all is not None and not dfR_all.empty:
        logger.info("\n" + "="*15 + " POST-RUN ANALYSIS " + "="*15)
        training_rounds = CONFIG.get('num_training_rounds', 0)

        # --- Evaluation Phase Analysis ---
        logger.info(f"Filtering for evaluation (Rounds >= {training_rounds})...")
        dfR_eval = dfR_all[dfR_all['round'] >= training_rounds].copy()
        logger.info(f"Found {len(dfR_eval)} evaluation rounds in round stats.")

        dfLogs_eval = pd.DataFrame() # Initialize empty
        if dfLogs_all is not None and not dfLogs_all.empty and 'round' in dfLogs_all.columns:
            try:
                 # Ensure 'round' is numeric before comparison
                 dfLogs_all['round'] = pd.to_numeric(dfLogs_all['round'], errors='coerce')
                 dfLogs_eval = dfLogs_all[dfLogs_all['round'] >= training_rounds].copy()
                 logger.info(f"Filtered {len(dfLogs_eval)} evaluation steps.")
            except Exception as e:
                logger.error(f"Could not filter step logs for evaluation analysis: {e}")
                dfLogs_eval = pd.DataFrame() # Ensure it's empty on error
        elif dfLogs_all is None or dfLogs_all.empty:
             logger.warning("No step logs available for evaluation analysis.")
        elif 'round' not in dfLogs_all.columns:
             logger.error("Step log DataFrame is missing 'round' column, cannot filter.")
             dfLogs_eval = pd.DataFrame()

        # Proceed with analysis only if evaluation rounds exist
        if dfR_eval.empty:
            logger.warning("No evaluation rounds found in round stats. Skipping evaluation analysis/plots.")
        else:
            # Convert eval rounds data to list of dicts for analysis functions
            # (Safer than passing DataFrame directly if functions expect list of dicts)
            eval_stats_list = dfR_eval.to_dict('records')

            try: # Individual Perf (Eval)
                ind_table = analyze_individual_performance(eval_stats_list)
                logger.info("\n" + ind_table)
            except Exception as e: logger.error(f"Error during Individual Analysis (EVAL): {e}", exc_info=True)

            try: # Market Perf (Eval)
                mkt_table = analyze_market_performance(eval_stats_list)
                logger.info("\n" + mkt_table)
            except Exception as e: logger.error(f"Error during Market Analysis (EVAL): {e}", exc_info=True)

            # --- ADDED CALL TO TOURNAMENT ANALYSIS ---
            try: # Strategy Tournament Ranking (Eval)
                tournament_table = analyze_strategy_tournament(eval_stats_list)
                logger.info("\n" + tournament_table)
            except Exception as e: logger.error(f"Error during Strategy Tournament Analysis (EVAL): {e}", exc_info=True)
            # --- END ADDED CALL ---

            # --- Evaluation Plotting ---
            logger.info("\nGenerating evaluation plots...")
            try:
                # Check plotting flags from final CONFIG
                generate_round_plots = CONFIG.get('generate_per_round_plots', False)
                generate_behavior_plots = CONFIG.get('generate_eval_behavior_plots', True) # Default True if RL agents might be present

                if generate_round_plots:
                    if dfLogs_eval.empty:
                        logger.warning("Skipping per-round plots: step log data for evaluation is missing or empty.")
                    else:
                        plot_per_round(eval_stats_list, exp_path, dfLogs=dfLogs_eval, generate_plots=True) # Pass generate_plots flag

                # Always generate summary plot for eval rounds if dfR_eval exists
                plot_game_summary(dfR_eval, exp_path, dfLogs=dfLogs_eval)

                if generate_behavior_plots:
                     if dfLogs_eval.empty:
                          logger.warning("Skipping RL behavior plots: step log data for evaluation is missing or empty.")
                     else:
                          plot_rl_behavior_eval(dfR_eval, dfLogs_eval, CONFIG, exp_path, num_rounds_to_plot=CONFIG.get('num_eval_rounds_to_plot', 5))

                logger.info("Evaluation plots generation attempted.")
            except Exception as e: logger.error(f"Error during plot generation (EVAL): {e}", exc_info=True)

        # --- Training Phase Plots ---
        if rl_training_logs:
             logger.info("Generating PPO training curves plot...")
             try:
                 plot_ppo_training_curves(rl_training_logs, exp_path)
             except Exception as e: logger.error(f"Error generating PPO training curves plot: {e}", exc_info=True)
        else:
             logger.info("No RL training logs found, skipping PPO training curves plot.")

    elif not run_successful:
        logger.warning("Skipping analysis and plotting due to auction run failure.")
    else: # Run successful but no data (dfR_all is None or empty)
        logger.warning("Skipping analysis and plotting due to missing or empty round data.")

    # --- Cleanup ---
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Experiment '{CONFIG['experiment_name']}' finished in {duration:.2f} seconds.")
    logging.shutdown() # Flush and close handlers
    print(f"\nExperiment finished. Results saved in: {exp_path}")
    if CONFIG.get('log_to_file', True) and log_file_path and os.path.exists(log_file_path):
        print(f"Check log file for details: {log_file_path}")
    elif CONFIG.get('log_to_file', True):
         print(f"Log file configured but not found at: {log_file_path}")


if __name__ == "__main__":
    # Ensure the script can find local modules when run directly
    # Assumes auction.py, utils.py, traders/ are in the same dir or PYTHONPATH
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    main()