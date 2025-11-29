#!/bin/bash
# Monitor PPO Training Progress
# Run in separate terminal: ./scripts/monitor_training.sh

LOGS_DIR="logs/ppo_training"
CHECKPOINTS_DIR="checkpoints"

echo "======================================================================"
echo "PPO TRAINING MONITOR"
echo "======================================================================"
echo "Monitoring logs in: $LOGS_DIR"
echo "Press Ctrl+C to exit"
echo ""

while true; do
    clear
    echo "======================================================================"
    echo "PPO TRAINING STATUS - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "======================================================================"
    echo ""

    # Check for log files
    if [ -d "$LOGS_DIR" ] && [ "$(ls -A $LOGS_DIR 2>/dev/null)" ]; then
        echo "📋 Active Training Logs:"
        echo "----------------------------------------------------------------------"
        ls -lh "$LOGS_DIR"/*.log 2>/dev/null | tail -5
        echo ""

        # Get latest log file
        LATEST_LOG=$(ls -t "$LOGS_DIR"/*.log 2>/dev/null | head -1)

        if [ -n "$LATEST_LOG" ]; then
            echo "📊 Latest Log: $(basename $LATEST_LOG)"
            echo "----------------------------------------------------------------------"
            tail -30 "$LATEST_LOG"
        fi
    else
        echo "⏳ No training logs found yet. Waiting for training to start..."
    fi

    echo ""
    echo "======================================================================"

    # Check checkpoints
    if [ -d "$CHECKPOINTS_DIR" ]; then
        echo "💾 Recent Checkpoints:"
        echo "----------------------------------------------------------------------"
        find "$CHECKPOINTS_DIR" -name "*.zip" -mmin -60 2>/dev/null | tail -5
    fi

    echo ""
    echo "Refreshing in 30 seconds... (Press Ctrl+C to stop)"
    sleep 30
done
