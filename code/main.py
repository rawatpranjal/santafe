import os, sys, shutil
import pprint
import pandas as pd

from config import CONFIG
from auction import Auction
from utils import (
    analyze_individual_performance,
    analyze_market_performance,
    plot_per_round,
    plot_game_summary
)
from tabulate import tabulate

def main():
    exp_path = os.path.join(CONFIG["experiment_dir"], CONFIG["experiment_name"])
    os.makedirs(exp_path, exist_ok=True)

    # Copy config
    shutil.copy("config.py", os.path.join(exp_path, "config.py"))

    log_file_path = os.path.join(exp_path, "stdout.log")
    with open(log_file_path, "w") as log_file:
        original_stdout = sys.stdout
        sys.stdout = log_file

        print("=== CONFIG SETTINGS ===")
        pprint.pprint(CONFIG)
        print("")

        # Run Auction
        auction = Auction(CONFIG)
        auction.run_auction()

        # Round-level => "round_log.csv"
        dfR = pd.DataFrame(auction.round_stats)
        round_log_path = os.path.join(exp_path, "round_log.csv")
        dfR.to_csv(round_log_path, index=False)
        print(f"\nSaved round stats to {round_log_path}")

        # Step-level => "log.csv"
        dfLogs = pd.DataFrame(auction.all_step_logs)
        log_csv_path = os.path.join(exp_path, "log.csv")
        dfLogs.to_csv(log_csv_path, index=False)
        print(f"Saved step-by-step logs to {log_csv_path}")

        # Print step logs for round=0
        df_r0 = dfLogs[dfLogs["round"] == 0].sort_values("step")
        print("\n=== STEP LOGS FOR ROUND=0 ===")
        df_r0["bids"] = df_r0["bids"].apply(lambda x: str(x))
        df_r0["asks"] = df_r0["asks"].apply(lambda x: str(x))
        df_r0["bprofits"] = df_r0["bprofits"].apply(lambda x: str(x))
        df_r0["sprofits"] = df_r0["sprofits"].apply(lambda x: str(x))
        step_table = tabulate(df_r0, headers="keys", tablefmt="pretty")
        print(step_table)

        # Analyze
        analyze_individual_performance(auction.round_stats)
        analyze_market_performance(auction.round_stats)

        # Plots (per-round & summary)
        plot_per_round(auction.round_stats, exp_path, dfLogs=dfLogs)
        plot_game_summary(dfR, exp_path, dfLogs=dfLogs)

        sys.stdout = original_stdout
    print(f"Experiment data saved in: {exp_path}")


if __name__ == "__main__":
    main()
