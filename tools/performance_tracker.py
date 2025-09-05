#!/usr/bin/env python3
"""
Neutral Performance Tracking Script

Analyzes PPO training logs to provide objective metrics:
- Rolling average profits over time windows
- Efficiency trends
- Success rate measurements
- Market competitiveness indicators

Usage: python performance_tracker.py <log_file>
"""

import sys
import re
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, Optional, Tuple

class PerformanceTracker:
    def __init__(self, window_size: int = 50):
        """
        Initialize performance tracker.
        
        Args:
            window_size: Rolling window size for average calculations
        """
        self.window_size = window_size
        self.data = []
        
    def parse_log_file(self, log_path: str) -> None:
        """Parse training log file to extract performance metrics."""
        print(f"Parsing log file: {log_path}")
        
        if not Path(log_path).exists():
            print(f"Error: Log file {log_path} not found")
            return
            
        with open(log_path, 'r') as f:
            content = f.read()
            
        # Extract round-by-round performance data from progress output
        # Format: Auction Rounds:  20%|██        | 118/600 [00:38<02:37,  3.07round/s, Eff=0.979, Mode=T, RLProfit=31892.0, AvgPL=-0.006, AvgVL=20.104]
        round_pattern = r'(\d+)/\d+ \[.*?Eff=([\d.]+), Mode=T, RLProfit=([\d.]+)'
        matches = re.findall(round_pattern, content)
        
        if not matches:
            # Fallback to alternative format
            round_pattern = r'Round (\d+).*?Efficiency: ([\d.]+)%.*?Total profit: (\d+)'
            matches = re.findall(round_pattern, content, re.DOTALL)
        
        for match in matches:
            round_num, efficiency, total_profit = match
            # Convert efficiency from decimal (0.979) to percentage (97.9%)
            eff_pct = float(efficiency) * 100 if float(efficiency) <= 1.0 else float(efficiency)
            self.data.append({
                'round': int(round_num),
                'efficiency': eff_pct,
                'total_profit': int(float(total_profit))  # Handle decimal format
            })
            
        # Extract PPO-specific metrics if available
        ppo_pattern = r'PPO agents.*?profit: (\d+)'
        ppo_matches = re.findall(ppo_pattern, content)
        
        if ppo_matches and len(ppo_matches) == len(self.data):
            for i, ppo_profit in enumerate(ppo_matches):
                self.data[i]['ppo_profit'] = int(ppo_profit)
                
        print(f"Extracted {len(self.data)} rounds of data")
        
    def calculate_rolling_metrics(self) -> pd.DataFrame:
        """Calculate rolling average metrics."""
        if not self.data:
            return pd.DataFrame()
            
        df = pd.DataFrame(self.data)
        
        # Rolling averages
        df['profit_rolling_avg'] = df['total_profit'].rolling(
            window=self.window_size, min_periods=1
        ).mean()
        
        df['efficiency_rolling_avg'] = df['efficiency'].rolling(
            window=self.window_size, min_periods=1
        ).mean()
        
        # Success rate (efficiency > 80%)
        df['success'] = (df['efficiency'] > 80.0).astype(int)
        df['success_rate'] = df['success'].rolling(
            window=self.window_size, min_periods=1
        ).mean() * 100
        
        # PPO-specific metrics if available
        if 'ppo_profit' in df.columns:
            df['ppo_profit_rolling_avg'] = df['ppo_profit'].rolling(
                window=self.window_size, min_periods=1
            ).mean()
            
            # PPO profit share
            df['ppo_profit_share'] = (df['ppo_profit'] / df['total_profit'] * 100).fillna(0)
            df['ppo_profit_share_rolling'] = df['ppo_profit_share'].rolling(
                window=self.window_size, min_periods=1
            ).mean()
        
        return df
        
    def generate_report(self, df: pd.DataFrame) -> str:
        """Generate neutral performance report."""
        if df.empty:
            return "No data available for analysis."
            
        report = []
        report.append("=== PPO PERFORMANCE ANALYSIS ===\n")
        
        # Overall statistics
        report.append("OVERALL STATISTICS:")
        report.append(f"Total rounds analyzed: {len(df)}")
        report.append(f"Average efficiency: {df['efficiency'].mean():.1f}%")
        report.append(f"Average total profit: {df['total_profit'].mean():.0f}")
        
        if 'ppo_profit' in df.columns:
            report.append(f"Average PPO profit: {df['ppo_profit'].mean():.0f}")
            report.append(f"PPO profit share: {df['ppo_profit_share'].mean():.1f}%")
        
        report.append("")
        
        # Trend analysis (first vs last 25% of data)
        n_early = len(df) // 4
        n_late = len(df) // 4
        
        early_data = df.iloc[:n_early]
        late_data = df.iloc[-n_late:]
        
        report.append("TREND ANALYSIS (Early 25% vs Late 25%):")
        
        early_eff = early_data['efficiency'].mean()
        late_eff = late_data['efficiency'].mean()
        eff_change = late_eff - early_eff
        
        report.append(f"Efficiency: {early_eff:.1f}% → {late_eff:.1f}% ({eff_change:+.1f}%)")
        
        early_profit = early_data['total_profit'].mean()
        late_profit = late_data['total_profit'].mean()
        profit_change = late_profit - early_profit
        
        report.append(f"Total profit: {early_profit:.0f} → {late_profit:.0f} ({profit_change:+.0f})")
        
        if 'ppo_profit' in df.columns:
            early_ppo = early_data['ppo_profit'].mean()
            late_ppo = late_data['ppo_profit'].mean()
            ppo_change = late_ppo - early_ppo
            
            report.append(f"PPO profit: {early_ppo:.0f} → {late_ppo:.0f} ({ppo_change:+.0f})")
        
        report.append("")
        
        # Performance milestones
        report.append("PERFORMANCE MILESTONES:")
        high_eff_rounds = df[df['efficiency'] > 90].index
        if len(high_eff_rounds) > 0:
            first_90 = high_eff_rounds[0] + 1
            report.append(f"First 90%+ efficiency: Round {first_90}")
        
        consistent_rounds = df[df['efficiency_rolling_avg'] > 85].index
        if len(consistent_rounds) > 0:
            first_consistent = consistent_rounds[0] + 1
            report.append(f"First consistent 85%+ efficiency (rolling): Round {first_consistent}")
            
        # Latest performance window
        report.append("")
        report.append(f"LATEST PERFORMANCE (Last {self.window_size} rounds):")
        if len(df) >= self.window_size:
            latest = df.iloc[-self.window_size:]
            report.append(f"Average efficiency: {latest['efficiency'].mean():.1f}%")
            report.append(f"Average total profit: {latest['total_profit'].mean():.0f}")
            report.append(f"Success rate (>80% eff): {latest['success'].mean()*100:.1f}%")
            
            if 'ppo_profit' in df.columns:
                report.append(f"Average PPO profit: {latest['ppo_profit'].mean():.0f}")
                report.append(f"PPO profit share: {latest['ppo_profit_share'].mean():.1f}%")
        
        return "\n".join(report)
        
    def save_detailed_csv(self, df: pd.DataFrame, output_path: str) -> None:
        """Save detailed metrics to CSV for further analysis."""
        if df.empty:
            print("No data to save")
            return
            
        df.to_csv(output_path, index=False)
        print(f"Detailed metrics saved to: {output_path}")
        
def main():
    parser = argparse.ArgumentParser(description="Analyze PPO training performance")
    parser.add_argument("log_file", help="Path to training log file")
    parser.add_argument("--window", type=int, default=50, 
                       help="Rolling window size (default: 50)")
    parser.add_argument("--output", help="Output CSV path for detailed metrics")
    
    args = parser.parse_args()
    
    tracker = PerformanceTracker(window_size=args.window)
    tracker.parse_log_file(args.log_file)
    
    df = tracker.calculate_rolling_metrics()
    report = tracker.generate_report(df)
    
    print(report)
    
    if args.output:
        tracker.save_detailed_csv(df, args.output)
        
if __name__ == "__main__":
    main()