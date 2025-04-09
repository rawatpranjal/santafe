# run_experiments.py
import subprocess
import os
import sys
import time

# --- Configuration ---
PYTHON_EXE = sys.executable # Use the python interpreter that is running this script
MAIN_SCRIPT = "main.py"    # Path to your main simulation script (relative to this script)
CONFIG_DIR = "configs"     # Directory containing the configuration files
BASE_LOG_LEVEL = "INFO"    # Default log level for console unless overridden by specific args below
BASE_RL_LOG_LEVEL = "WARNING" # Default RL log level unless overridden

# --- Define Experiments ---
# List of configuration file names located within CONFIG_DIR
# Includes 01* (baseline ZI/ZIC/ZIP) and the new 02* (full fixed strategy mix) experiments
config_files_to_run = [
    # --- 01: Baseline Runs (ZI, ZIC, ZIP) ---
    #"01a_baseline_ziu_symmetric_1k.py",
    #"01b_baseline_zic_symmetric_1k.py",
    #"01c_baseline_zip_symmetric_1k.py", # Assuming this exists
    #"01d_baseline_zic_asymmetric_1k.py",
    #"01e_baseline_zip_asymmetric_1k.py", # Assuming this exists

    # --- 02: Full Fixed Strategy Mix Runs ---
    "02a_fixed_strategy_mix_baseline_1k.py",
    "02b_fixed_strategy_mix_more_tokens_1k.py",
    "02c_fixed_strategy_mix_double_agents_1k.py",
    "02d_fixed_strategy_mix_more_time_1k.py",

    # --- 03: Add MARL LSTM runs later when ready ---
    # "config_03a_marl_base_lstm_5k.py",
    # "config_03b_marl_openbook_lstm_5k.py",
    # "config_03c_marl_minimal_lstm_5k.py",
]

# --- Experiment Execution Loop ---
total_start_time = time.time()
failed_experiments = []
successful_experiments = []

# Ensure the script directory is the working directory if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
# os.chdir(script_dir) # Uncomment if main.py relies on running from its own directory

print(f"Starting experiment runner...")
print(f"Found {len(config_files_to_run)} experiments to run.")
print(f"Using Python: {PYTHON_EXE}")
print(f"Main script: {os.path.join(script_dir, MAIN_SCRIPT)}")
print(f"Config directory: {os.path.join(script_dir, CONFIG_DIR)}")
print("-" * 60)

for i, config_filename in enumerate(config_files_to_run):
    exp_num = i + 1
    # Construct the full path relative to the script's directory
    config_filepath = os.path.join(script_dir, CONFIG_DIR, config_filename)
    # Derive experiment name from config filename (strip .py)
    # This will be passed via --name to override the name inside the config file
    # Ensures the experiment folder name matches the config file name
    experiment_name_override = os.path.splitext(config_filename)[0]

    print(f"\n{'='*20} Running Experiment {exp_num}/{len(config_files_to_run)} {'='*20}")
    print(f"Config File: {config_filename}")
    print(f"Full Path:   {config_filepath}")
    print(f"Output Name: {experiment_name_override}")

    # Check if config file exists before attempting to run
    if not os.path.exists(config_filepath):
        print(f"!!!!!! ERROR: Config file not found: {config_filepath} !!!!!!")
        failed_experiments.append(f"{experiment_name_override} (Config Not Found)")
        continue # Skip to the next experiment

    # Construct the command line arguments for main.py
    command = [
        PYTHON_EXE,                            # e.g., /usr/bin/python3
        os.path.join(script_dir, MAIN_SCRIPT), # Full path to main.py
        "--config_file", config_filepath,      # Pass full path to config
        "--name", experiment_name_override,    # Ensure output folder matches config name
        "--log", BASE_LOG_LEVEL,               # Set console log level
        "--log_rl", BASE_RL_LOG_LEVEL,         # Set RL specific log level
        # Add any other flags you want applied to ALL runs by default
        # "--no_round_plots", # Example: uncomment to disable for all runs
        # "--no_eval_plots",  # Example: uncomment to disable for all runs
    ]

    print(f"Executing Command: {' '.join(command)}")
    print("-" * 40)
    exp_start_time = time.time()

    try:
        # Run main.py as a subprocess
        # stdout and stderr will be printed directly to the console where run_experiments is running
        result = subprocess.run(command, check=True, text=True, encoding='utf-8', stdout=sys.stdout, stderr=sys.stderr)
        exp_duration = time.time() - exp_start_time
        print(f"----- Experiment {exp_num} ({experiment_name_override}) Finished Successfully ({exp_duration:.2f}s) -----")
        successful_experiments.append(experiment_name_override)

    except subprocess.CalledProcessError as e:
        exp_duration = time.time() - exp_start_time
        print(f"!!!!!! Experiment {exp_num} ({experiment_name_override}) FAILED after {exp_duration:.2f}s !!!!!!")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Return Code: {e.returncode}")
        # Error output should have already printed to stderr
        failed_experiments.append(f"{experiment_name_override} (Runtime Error: Code {e.returncode})")
        # Optional: stop execution if one fails
        # print("Stopping runner due to error.")
        # break
    except FileNotFoundError:
         print(f"!!!!!! ERROR: Could not find python '{PYTHON_EXE}' or script '{MAIN_SCRIPT}'. Check paths. !!!!!!")
         failed_experiments.append(f"{experiment_name_override} (File Not Found Error)")
         print("Stopping runner.")
         break # Stop if python/script not found
    except Exception as e:
         exp_duration = time.time() - exp_start_time
         print(f"!!!!!! An unexpected error occurred running experiment {exp_num} ({experiment_name_override}) after {exp_duration:.2f}s: {e} !!!!!!")
         failed_experiments.append(f"{experiment_name_override} (Unexpected Error: {type(e).__name__})")
         # Optional: stop execution
         # break
    finally:
        # Add a small delay between runs if desired (e.g., to let disk I/O settle)
        # time.sleep(1)
        pass

# --- Final Summary ---
total_duration = time.time() - total_start_time
print(f"\n{'='*20} All Experiments Attempted ({total_duration:.2f}s) {'='*20}")
print(f"Successful: {len(successful_experiments)}")
print(f"Failed:     {len(failed_experiments)}")
if failed_experiments:
    print("\nFailed experiments:")
    for failure in failed_experiments:
        print(f"  - {failure}")
else:
    print("\nAll experiments completed successfully!")
print("=" * (43 + len(f"({total_duration:.2f}s)"))) # Adjust width dynamically

# Optional: Exit with non-zero code if any experiments failed
if failed_experiments:
    sys.exit(1)
else:
    sys.exit(0)