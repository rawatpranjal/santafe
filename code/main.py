# main.py
import os
import sys
import shutil
import pprint
import pandas as pd
import logging
import argparse
# from tqdm import tqdm # Not needed directly in main

# Make sure local modules can be imported if running script directly
# script_dir = os.path.dirname(__file__)
# code_dir = os.path.join(script_dir, '..', 'code') # Adjust relative path if needed
# if code_dir not in sys.path:
#     sys.path.append(code_dir)

from config import CONFIG
from auction import Auction
from utils import ( # Ensure utils has the latest analysis/plotting
    analyze_individual_performance,
    analyze_market_performance,
    plot_per_round,
    plot_game_summary,
    plot_dqn_behavior_eval # Import the new function
)
from traders.registry import available_strategies # Import list of strategies
# Import Kaplan explicitly if needed elsewhere, though registry handles it
# from traders.kaplan import KaplanBuyer, KaplanSeller

# --- Argument Parsing (Added RL log level and plot flags) ---
parser = argparse.ArgumentParser(description="Run SFI Double Auction Simulation")
parser.add_argument('--log', default=CONFIG.get('log_level', 'INFO').upper(),
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Set the default logging level')
parser.add_argument('--log_rl', default=CONFIG.get('log_level_rl', 'WARNING').upper(),
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Set the RL agent logging level')
parser.add_argument('--name', default=CONFIG['experiment_name'], help='Override experiment name from config')
parser.add_argument('--train_rounds', type=int, default=CONFIG.get('num_training_rounds'), help='Override number of RL training rounds')
parser.add_argument('--load_model', default=CONFIG.get('load_rl_model_path'), help='Path to load pre-trained RL model state dict')
parser.add_argument('--no_save', action='store_true', help="Disable saving trained RL model")
parser.add_argument('--no_round_plots', action='store_true', help="Disable generation of per-round plots")
parser.add_argument('--no_eval_plots', action='store_true', help="Disable generation of DQN evaluation behavior plots") # New flag

args = parser.parse_args()

# Override config values if provided via CLI
CONFIG['log_level'] = args.log
CONFIG['log_level_rl'] = args.log_rl # Store RL log level
CONFIG['experiment_name'] = args.name
CONFIG['num_training_rounds'] = args.train_rounds
CONFIG['load_rl_model_path'] = args.load_model
if args.no_save: CONFIG['save_rl_model'] = False
if args.no_round_plots: CONFIG['generate_per_round_plots'] = False
if args.no_eval_plots: CONFIG['generate_eval_behavior_plots'] = False # Override eval plot flag

# --- Logging Setup ---
log_level_default = getattr(logging, CONFIG.get('log_level', 'INFO').upper(), logging.INFO)
log_level_rl = getattr(logging, CONFIG.get('log_level_rl', 'WARNING').upper(), logging.WARNING)
log_format = '%(asctime)s - %(levelname)-8s - %(name)-15s - %(message)s'
date_format = '%Y-%m-%d %H:%M:%S'

# Configure root logger
logging.basicConfig(level=log_level_default, format=log_format, datefmt=date_format, stream=sys.stdout, force=True)

# Set specific level for RL components based on config/args
rl_logger_names = ['rl_agent_logic.dqn'] # Add other RL logger names if needed
for name in rl_logger_names: logging.getLogger(name).setLevel(log_level_rl)
# Set level for individual trader logs based on default level (can be overridden if needed)
logging.getLogger('trader').setLevel(log_level_default) # Keep trader logs at default for now

logger = logging.getLogger('main') # Logger for this script

def main():
    # --- Experiment Setup ---
    exp_base_dir = CONFIG["experiment_dir"]
    exp_name = CONFIG["experiment_name"]
    exp_path = os.path.join(exp_base_dir, exp_name)

    # Ensure experiment directory exists
    try: os.makedirs(exp_path, exist_ok=True)
    except OSError as e: print(f"FATAL: Could not create experiment directory {exp_path}. Error: {e}"); sys.exit(1)

    # Configure file logging if enabled
    log_file_path = os.path.join(exp_path, "run.log")
    if CONFIG.get('log_to_file', True):
        try:
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]: root_logger.removeHandler(handler) # Remove default StreamHandler
            # File handler - logs everything at DEBUG level or higher
            file_handler = logging.FileHandler(log_file_path, mode='w')
            file_formatter = logging.Formatter(log_format, datefmt=date_format)
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG) # Log detailed info to file
            root_logger.addHandler(file_handler)
            # Console handler - logs at the user-specified default level
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter(log_format, datefmt=date_format) # Can use same format
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(log_level_default) # Use default level for console
            root_logger.addHandler(console_handler)
            # Set overall level for root logger (so handlers receive messages)
            root_logger.setLevel(logging.DEBUG) # Process all messages, let handlers filter
            print(f"Logging detailed output to: {log_file_path}")
            print(f"Console log level set to: {args.log}")
        except Exception as e:
            print(f"Warning: Could not set up file logging to {log_file_path}: {e}")
            # Fallback to basic console logging if file setup fails
            logging.basicConfig(level=log_level_default, format=log_format, datefmt=date_format, stream=sys.stdout, force=True)
            logging.getLogger('rl_agent_logic.dqn').setLevel(log_level_rl) # Re-apply RL level

    logger.info(f"Starting experiment: {exp_name}")
    logger.info(f"Results will be saved to: {exp_path}")
    logger.info(f"Available strategies in registry: {available_strategies()}")
    # Copy config file for reproducibility
    try: shutil.copy("config.py", os.path.join(exp_path, "config_run.py")) # Rename to avoid clash
    except Exception as e: logger.warning(f"Could not copy config.py: {e}")

    # Log the final configuration being used
    logger.info("=== FINAL CONFIGURATION ===")
    config_str = pprint.pformat(CONFIG, indent=2, width=100)
    for line in config_str.splitlines(): logger.info(line)
    logger.info("===========================")

    # --- Run Auction ---
    auction_instance = None
    try:
        auction_instance = Auction(CONFIG)
        auction_instance.run_auction()
        logger.info("Auction run completed successfully.")
    except Exception as e:
        logger.exception("FATAL ERROR during auction run!")
        logging.shutdown()
        sys.exit(1)

    # --- Save Results ---
    dfR, dfLogs = None, None
    if auction_instance and auction_instance.round_stats:
        try:
            dfR = pd.DataFrame(auction_instance.round_stats)
            # Convert complex columns to string for CSV compatibility
            for col in ['buyer_vals', 'seller_vals', 'role_strat_perf', 'bot_details']:
                 if col in dfR.columns: dfR[col] = dfR[col].astype(str)
            round_log_path = os.path.join(exp_path, "round_log.csv")
            dfR.to_csv(round_log_path, index=False)
            logger.info(f"Saved round stats to {round_log_path}")
        except Exception as e: logger.error(f"Error saving round_log.csv: {e}", exc_info=True); dfR = None
    else: logger.warning("No round stats generated by auction.")

    if auction_instance and auction_instance.all_step_logs:
        try:
            dfLogs = pd.DataFrame(auction_instance.all_step_logs)
            # Convert dict columns to string
            for col in ['bids_submitted', 'asks_submitted']:
                 if col in dfLogs.columns: dfLogs[col] = dfLogs[col].astype(str)
            log_csv_path = os.path.join(exp_path, "step_log.csv")
            dfLogs.to_csv(log_csv_path, index=False)
            logger.info(f"Saved step-by-step logs to {log_csv_path}")
        except Exception as e: logger.error(f"Error saving step_log.csv: {e}", exc_info=True); dfLogs = None
    else: logger.warning("No step logs generated by auction.")

    # --- Analysis & Plotting ---
    if dfR is not None and not dfR.empty:
        logger.info("\n" + "="*20 + " ANALYSIS RESULTS " + "="*20)
        try:
            individual_perf_table = analyze_individual_performance(auction_instance.round_stats)
            # Log table to logger (which might go to file and/or console)
            for line in individual_perf_table.splitlines(): logger.info(line)
        except Exception as e:
            logger.error(f"Error analyzing individual performance: {e}", exc_info=True)

        try:
            market_perf_table = analyze_market_performance(auction_instance.round_stats)
            for line in market_perf_table.splitlines(): logger.info(line)
        except Exception as e:
            logger.error(f"Error analyzing market performance: {e}", exc_info=True)

        logger.info("\nGenerating plots...")
        try:
            # Plot per-round (optional based on config)
            plot_per_round(
                auction_instance.round_stats,
                exp_path,
                dfLogs=(dfLogs if dfLogs is not None else pd.DataFrame()),
                generate_plots=CONFIG.get('generate_per_round_plots', True)
            )
            # Plot overall summary
            plot_game_summary(
                dfR,
                exp_path,
                dfLogs=(dfLogs if dfLogs is not None else pd.DataFrame())
            )
            # Plot DQN Eval Behavior (optional based on config)
            if CONFIG.get('generate_eval_behavior_plots', True):
                # <<< Corrected call here >>>
                plot_dqn_behavior_eval(
                    dfR,
                    (dfLogs if dfLogs is not None else pd.DataFrame()), # Pass dfLogs positionally
                    CONFIG, # Pass config
                    exp_path, # Pass path
                    num_rounds_to_plot=CONFIG.get('num_eval_rounds_to_plot', 5) # Pass num rounds
                )
            logger.info("Plots generation attempted.")
        except Exception as e:
            logger.error(f"Error during plot generation: {e}", exc_info=True)
    else: logger.warning("Skipping analysis and plotting due to missing/failed round data.")

    # --- Cleanup ---
    logger.info("Main script finished.")
    logging.shutdown() # Ensure all handlers are flushed/closed
    print(f"\nExperiment finished. Results saved in: {exp_path}")
    if CONFIG.get('log_to_file', True): print(f"Check log file for details: {log_file_path}")

if __name__ == "__main__":
    main()