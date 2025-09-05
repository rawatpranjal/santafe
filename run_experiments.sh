#!/bin/bash

# run_experiments.sh - Automated experiment runner for Santa Fe Double Auction
# 
# This script provides one-click experimental runs with automated report generation
# Runs comprehensive strategy tournaments and generates research-quality reports

set -e  # Exit on any error

echo "============================================================"
echo "Santa Fe Double Auction - Automated Experiment Suite"
echo "============================================================"
echo ""

# Check if we're in the correct directory
if [ ! -f "src_code/main.py" ] && [ ! -f "code/main.py" ]; then
    echo "Error: Please run this script from the repository root directory"
    exit 1
fi

# Set up virtual environment if needed
USING_VENV=false
if ! python3 -c "import numpy, pandas, matplotlib" 2>/dev/null; then
    echo "⚠️  Required packages not found. Setting up virtual environment..."
    
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    USING_VENV=true
    
    if [ ! -f "venv/.requirements_installed" ]; then
        echo "Installing requirements..."
        pip install -r requirements.txt
        touch venv/.requirements_installed
    fi
    
    echo "✅ Environment ready"
else
    echo "✅ Required packages available"
fi

# Set python command
PYTHON_CMD="python3"
if [ "$USING_VENV" = true ]; then
    PYTHON_CMD="python"
fi

# Create results directory
mkdir -p results/experiments
mkdir -p results/reports

echo ""
echo "🧪 Running Santa Fe Strategy Tournament..."
echo "----------------------------------------"

# Strategy tournament configuration
STRATEGIES=("zic" "zip" "mgd")
NUM_ROUNDS=10
NUM_PERIODS=5
NUM_STEPS=100

echo "Strategies: ${STRATEGIES[@]}"
echo "Configuration: ${NUM_ROUNDS} rounds, ${NUM_PERIODS} periods, ${NUM_STEPS} steps per period"
echo ""

# Run pairwise strategy comparisons
echo "🥊 Running pairwise strategy tournaments..."
for buyer_strategy in "${STRATEGIES[@]}"; do
    for seller_strategy in "${STRATEGIES[@]}"; do
        echo "  Running: ${buyer_strategy} vs ${seller_strategy}..."
        
        $PYTHON_CMD -c "
import sys
import os
import json
import pandas as pd
from datetime import datetime
sys.path.insert(0, 'src_code')

from auction import Auction
from traders.registry import get_trader_class

# Configuration
config = {
    'experiment_name': f'tournament_${buyer_strategy}_vs_${seller_strategy}',
    'num_rounds': ${NUM_ROUNDS},
    'num_periods': ${NUM_PERIODS},
    'num_steps': ${NUM_STEPS},
    'num_buyers': 3,
    'num_sellers': 3,
    'num_tokens': 5,
    'min_price': 1,
    'max_price': 1000,
    'gametype': 6453,  # Random values
    'buyers': [{'class': get_trader_class('${buyer_strategy}', is_buyer=True)}] * 3,
    'sellers': [{'class': get_trader_class('${seller_strategy}', is_buyer=False)}] * 3,
    'rng_seed_auction': 42,
    'rng_seed_values': 123,
}

# Run auction
auction = Auction(config)
auction.run_auction()

# Save results
results = {
    'config': config,
    'round_stats': auction.round_stats,
    'step_logs': auction.all_step_logs,
    'timestamp': datetime.now().isoformat()
}

# Calculate summary statistics
avg_efficiency = sum(r['market_efficiency'] for r in auction.round_stats) / len(auction.round_stats)
avg_trades = sum(r['actual_trades'] for r in auction.round_stats) / len(auction.round_stats)
total_trades = sum(r['actual_trades'] for r in auction.round_stats)

results['summary'] = {
    'buyer_strategy': '${buyer_strategy}',
    'seller_strategy': '${seller_strategy}',
    'avg_market_efficiency': avg_efficiency,
    'avg_trades_per_round': avg_trades,
    'total_trades': total_trades
}

# Save to file
filename = f'results/experiments/tournament_${buyer_strategy}_vs_${seller_strategy}.json'
with open(filename, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f'✅ Completed: ${buyer_strategy} vs ${seller_strategy} - Efficiency: {avg_efficiency:.3f}, Trades: {total_trades}')
"
    done
done

echo ""
echo "📊 Generating comprehensive tournament report..."
echo "----------------------------------------------"

# Generate tournament report
$PYTHON_CMD -c "
import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
sys.path.insert(0, 'src_code')

# Load all experiment results
results_dir = 'results/experiments'
all_results = []

for filename in os.listdir(results_dir):
    if filename.endswith('.json') and 'tournament_' in filename:
        with open(os.path.join(results_dir, filename), 'r') as f:
            data = json.load(f)
            all_results.append(data['summary'])

if not all_results:
    print('No experiment results found!')
    sys.exit(1)

# Create results DataFrame
df = pd.DataFrame(all_results)

print('🏆 Santa Fe Strategy Tournament Results')
print('=' * 50)
print()

# Overall strategy performance
print('📈 Market Efficiency by Strategy Pair:')
print(df[['buyer_strategy', 'seller_strategy', 'avg_market_efficiency']].to_string(index=False))
print()

print('🔄 Total Trades by Strategy Pair:')
print(df[['buyer_strategy', 'seller_strategy', 'total_trades']].to_string(index=False))
print()

# Strategy rankings
buyer_performance = df.groupby('buyer_strategy')['avg_market_efficiency'].mean().sort_values(ascending=False)
seller_performance = df.groupby('seller_strategy')['avg_market_efficiency'].mean().sort_values(ascending=False)

print('🥇 Buyer Strategy Rankings (by avg efficiency):')
for i, (strategy, efficiency) in enumerate(buyer_performance.items(), 1):
    print(f'  {i}. {strategy.upper()}: {efficiency:.3f}')
print()

print('🥈 Seller Strategy Rankings (by avg efficiency):')
for i, (strategy, efficiency) in enumerate(seller_performance.items(), 1):
    print(f'  {i}. {strategy.upper()}: {efficiency:.3f}')
print()

# Create competition matrix
strategies = df['buyer_strategy'].unique()
matrix_data = []

for buyer in strategies:
    row = []
    for seller in strategies:
        match = df[(df['buyer_strategy'] == buyer) & (df['seller_strategy'] == seller)]
        if len(match) > 0:
            row.append(match.iloc[0]['avg_market_efficiency'])
        else:
            row.append(0.0)
    matrix_data.append(row)

competition_matrix = pd.DataFrame(matrix_data, index=strategies, columns=strategies)

print('🎯 Competition Matrix (Market Efficiency):')
print('   Rows = Buyers, Columns = Sellers')
print(competition_matrix.round(3).to_string())
print()

# Generate plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Strategy efficiency comparison
buyer_performance.plot(kind='bar', ax=ax1, color='skyblue')
ax1.set_title('Buyer Strategy Performance')
ax1.set_ylabel('Average Market Efficiency')
ax1.tick_params(axis='x', rotation=45)

seller_performance.plot(kind='bar', ax=ax2, color='lightcoral') 
ax2.set_title('Seller Strategy Performance')
ax2.set_ylabel('Average Market Efficiency')
ax2.tick_params(axis='x', rotation=45)

# Plot 3: Competition heatmap
im = ax3.imshow(competition_matrix.values, cmap='viridis', aspect='auto')
ax3.set_xticks(range(len(strategies)))
ax3.set_yticks(range(len(strategies)))
ax3.set_xticklabels([s.upper() for s in strategies])
ax3.set_yticklabels([s.upper() for s in strategies])
ax3.set_title('Strategy vs Strategy Efficiency Matrix')
ax3.set_xlabel('Seller Strategy')
ax3.set_ylabel('Buyer Strategy')

# Add values to heatmap
for i in range(len(strategies)):
    for j in range(len(strategies)):
        ax3.text(j, i, f'{competition_matrix.iloc[i, j]:.2f}', 
                ha='center', va='center', color='white', fontweight='bold')

# Plot 4: Trade volume comparison
trade_summary = df.groupby(['buyer_strategy', 'seller_strategy'])['total_trades'].sum()
trade_summary.plot(kind='bar', ax=ax4, color='gold')
ax4.set_title('Total Trades by Strategy Combination')
ax4.set_ylabel('Total Trades')
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('results/reports/tournament_analysis.png', dpi=300, bbox_inches='tight')
print('📊 Tournament analysis plot saved: results/reports/tournament_analysis.png')

# Generate LaTeX table for paper
latex_table = '''
\\\\begin{table}[h]
\\\\centering
\\\\caption{Santa Fe Strategy Tournament Results}
\\\\label{tab:tournament_results}
\\\\begin{tabular}{|l|c|c|c|}
\\\\hline
\\\\textbf{Strategy Pair} & \\\\textbf{Avg Efficiency} & \\\\textbf{Total Trades} & \\\\textbf{Trades/Round} \\\\\\\\
\\\\hline
'''

for _, row in df.iterrows():
    latex_table += f\"{row['buyer_strategy'].upper()} vs {row['seller_strategy'].upper()} & {row['avg_market_efficiency']:.3f} & {row['total_trades']} & {row['avg_trades_per_round']:.1f} \\\\\\\\\\\\\\n\"

latex_table += '''\\\\hline
\\\\end{tabular}
\\\\end{table}
'''

with open('results/reports/tournament_table.tex', 'w') as f:
    f.write(latex_table)
    
print('📄 LaTeX table saved: results/reports/tournament_table.tex')
print()

# Summary report
report_content = f'''
# Santa Fe Double Auction Tournament Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Experiment Configuration
- Strategies tested: {', '.join(strategies)}
- Rounds per matchup: ${NUM_ROUNDS}
- Periods per round: ${NUM_PERIODS}  
- Steps per period: ${NUM_STEPS}
- Traders per side: 3
- Tokens per trader: 5

## Key Findings
- Best performing buyer strategy: {buyer_performance.index[0].upper()} (efficiency: {buyer_performance.iloc[0]:.3f})
- Best performing seller strategy: {seller_performance.index[0].upper()} (efficiency: {seller_performance.iloc[0]:.3f})
- Highest efficiency matchup: {df.loc[df['avg_market_efficiency'].idxmax(), 'buyer_strategy'].upper()} vs {df.loc[df['avg_market_efficiency'].idxmax(), 'seller_strategy'].upper()} ({df['avg_market_efficiency'].max():.3f})
- Most active matchup: {df.loc[df['total_trades'].idxmax(), 'buyer_strategy'].upper()} vs {df.loc[df['total_trades'].idxmax(), 'seller_strategy'].upper()} ({df['total_trades'].max()} trades)

## Strategy Analysis
Each strategy demonstrates different characteristics:
- **ZIC (Zero Intelligence Constrained)**: Baseline random strategy within profit constraints
- **ZIP (Zero Intelligence Plus)**: Adaptive learning with Widrow-Hoff margin updates  
- **MGD (Modified Gjerstad-Dickhaut)**: Belief-based learning with historical price bounds

See tournament_analysis.png for detailed visualizations.
'''

with open('results/reports/tournament_report.md', 'w') as f:
    f.write(report_content)

print('📋 Full tournament report saved: results/reports/tournament_report.md')
print()
"

echo ""
echo "🎉 Experiment suite completed successfully!"
echo ""
echo "📁 Results saved in:"
echo "   - results/experiments/     (Raw experiment data)"
echo "   - results/reports/         (Analysis and reports)"
echo ""
echo "📊 Generated outputs:"
echo "   - tournament_analysis.png  (Comprehensive plots)"
echo "   - tournament_report.md     (Markdown report)"
echo "   - tournament_table.tex     (LaTeX table for papers)"
echo ""
echo "============================================================"