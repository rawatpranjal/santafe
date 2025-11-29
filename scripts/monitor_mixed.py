import re
import sys
import time
from datetime import datetime

log_file = "training_log_mixed_v3.txt"

def parse_log():
    try:
        with open(log_file, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Log file {log_file} not found.")
        return

    # Extract metrics using regex
    timesteps = re.findall(r"total_timesteps\s+\|\s+(\d+)", content)
    rewards = re.findall(r"(?:rollout/ep_rew_mean|eval/mean_reward)\s+\|\s+([\d\.\-]+)", content)
    trades = re.findall(r"trades\s+\|\s+([\d\.\-]+)", content)
    efficiency = re.findall(r"efficiency\s+\|\s+([\d\.\-]+)", content)
    profit = re.findall(r"profit\s+\|\s+([\d\.\-]+)", content)
    
    if not timesteps:
        print("No training data yet...")
        return

    print(f"\n=== Training Monitor ({datetime.now().strftime('%H:%M:%S')}) ===")
    print(f"Latest Timestep: {timesteps[-1]}")
    
    if rewards: print(f"Mean Reward:     {rewards[-1]}")
    if trades:  print(f"Trades/Ep:       {trades[-1]}")
    if efficiency: print(f"Efficiency:      {efficiency[-1]}")
    if profit:  print(f"Total Profit:    {profit[-1]}")
    
    # Show history of last 5 entries
    print("\nHistory (Last 5):")
    print(f"{'Step':<10} | {'Reward':<10} | {'Trades':<10} | {'Eff %':<10} | {'Profit':<10}")
    print("-" * 60)
    
    limit = min(len(timesteps), 5)
    for i in range(1, limit + 1):
        idx = -i
        t = timesteps[idx] if len(timesteps) >= i else "-"
        r = rewards[idx] if len(rewards) >= i else "-"
        tr = trades[idx] if len(trades) >= i else "-"
        e = efficiency[idx] if len(efficiency) >= i else "-"
        p = profit[idx] if len(profit) >= i else "-"
        print(f"{t:<10} | {r:<10} | {tr:<10} | {e:<10} | {p:<10}")

if __name__ == "__main__":
    parse_log()
