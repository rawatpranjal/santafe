#!/bin/bash
# Master PPO Training Script - Phase 4 Experiments
# Runs all 4 PPO training configurations sequentially
# Total estimated time: 12-24 hours

set -e  # Exit on error

echo "======================================================================"
echo "PHASE 4: PPO TRAINING EXPERIMENTS"
echo "======================================================================"
echo "Start time: $(date)"
echo ""
echo "This script will run 4 training configurations:"
echo "  1. PPO vs ZIC (1M steps, ~2-4 hours)"
echo "  2. PPO vs Kaplan (2M steps, ~4-6 hours)"
echo "  3. PPO vs Mixed (2M steps, ~4-6 hours)"
echo "  4. PPO Curriculum (3M steps, ~6-8 hours)"
echo ""
echo "Total estimated time: 16-24 hours"
echo "======================================================================"
echo ""

# Create results directory
mkdir -p results/ppo_training
mkdir -p logs/ppo_training

# Function to run training with error handling
run_training() {
    local config=$1
    local timesteps=$2
    local name=$3

    echo ""
    echo "======================================================================"
    echo "STARTING: $name"
    echo "Config: $config"
    echo "Timesteps: $timesteps"
    echo "Start time: $(date)"
    echo "======================================================================"

    # Run training
    if python scripts/train_ppo_enhanced.py \
        --config "$config" \
        --timesteps "$timesteps" \
        --no-wandb \
        --no-tensorboard \
        --verbose \
        2>&1 | tee "logs/ppo_training/${config}_$(date +%Y%m%d_%H%M%S).log"; then

        echo ""
        echo "✅ COMPLETED: $name"
        echo "End time: $(date)"
        echo ""
    else
        echo ""
        echo "❌ FAILED: $name"
        echo "Check log: logs/ppo_training/${config}_*.log"
        echo ""
        return 1
    fi
}

# Track overall progress
TOTAL_CONFIGS=4
COMPLETED=0

# 1. PPO vs ZIC (Easy baseline)
if run_training "ppo_vs_zic" 1000000 "PPO vs ZIC (Baseline)"; then
    COMPLETED=$((COMPLETED + 1))
fi

# 2. PPO vs Kaplan (Strategic challenge)
if run_training "ppo_vs_kaplan" 2000000 "PPO vs Kaplan (Strategic)"; then
    COMPLETED=$((COMPLETED + 1))
fi

# 3. PPO vs Mixed (Generalization test)
if run_training "ppo_vs_mixed" 2000000 "PPO vs Mixed (Realistic)"; then
    COMPLETED=$((COMPLETED + 1))
fi

# 4. PPO Curriculum (Progressive learning)
if run_training "ppo_curriculum" 3000000 "PPO Curriculum (Full)"; then
    COMPLETED=$((COMPLETED + 1))
fi

# Summary
echo ""
echo "======================================================================"
echo "TRAINING COMPLETE"
echo "======================================================================"
echo "End time: $(date)"
echo "Completed: $COMPLETED / $TOTAL_CONFIGS configurations"
echo ""

if [ $COMPLETED -eq $TOTAL_CONFIGS ]; then
    echo "✅ All training runs completed successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Run evaluation: ./scripts/evaluate_all_ppo.sh"
    echo "  2. Generate figures: python scripts/generate_paper_artifacts.py"
    echo "  3. Update paper: Edit paper/arxiv/sections/06_results_rl.tex"
else
    echo "⚠️  Some training runs failed. Check logs in logs/ppo_training/"
fi

echo "======================================================================"
