#!/bin/bash
# Live Tournament Dashboard
# Updates every 5 seconds with current status

while true; do
    clear
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║              SANTA FE TOURNAMENT EXECUTION DASHBOARD                       ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # Count completions
    PURE_DONE=$(grep -c "✓ Completed" tournament_pure.log 2>/dev/null || echo 0)
    PAIRWISE_DONE=$(grep -c "✓ Completed" tournament_pairwise.log 2>/dev/null || echo 0)
    MIXED_DONE=$(grep -c "✓ Completed" tournament_mixed.log 2>/dev/null || echo 0)
    ONESEVEN_DONE=$(grep -c "✓ Completed" tournament_1v7.log 2>/dev/null || echo 0)
    ASYMMETRIC_DONE=$(grep -c "✓ Completed" tournament_asymmetric.log 2>/dev/null || echo 0)

    TOTAL_DONE=$((PURE_DONE + PAIRWISE_DONE + MIXED_DONE + ONESEVEN_DONE + ASYMMETRIC_DONE))
    TOTAL=89
    PERCENT=$((TOTAL_DONE * 100 / TOTAL))

    echo "┌────────────────────────────────────────────────────────────────────────────┐"
    echo "│ OVERALL PROGRESS: $TOTAL_DONE / $TOTAL ($PERCENT%)                                        │"
    echo "└────────────────────────────────────────────────────────────────────────────┘"
    echo ""

    # Progress bars
    echo "┌─ TOURNAMENT CATEGORIES ─────────────────────────────────────────────────┐"
    echo "│"

    # Pure
    printf "│ 1. PURE SELF-PLAY      [$PURE_DONE/10]   "
    if [ $PURE_DONE -eq 10 ]; then
        echo "✅ COMPLETE"
    else
        echo "⏳ RUNNING"
    fi

    # Pairwise
    printf "│ 2. PAIRWISE            [$PAIRWISE_DONE/45]  "
    if [ $PAIRWISE_DONE -eq 45 ]; then
        echo "✅ COMPLETE"
    else
        echo "⏳ RUNNING"
    fi

    # Mixed
    printf "│ 3. MIXED STRATEGIES    [$MIXED_DONE/8]   "
    if [ $MIXED_DONE -eq 8 ]; then
        echo "✅ COMPLETE"
    else
        echo "⏳ RUNNING"
    fi

    # 1v7
    printf "│ 4. INVASIBILITY (1v7)  [$ONESEVEN_DONE/20]  "
    if [ $ONESEVEN_DONE -eq 20 ]; then
        echo "✅ COMPLETE"
    else
        echo "⏳ RUNNING"
    fi

    # Asymmetric
    printf "│ 5. ASYMMETRIC          [$ASYMMETRIC_DONE/6]   "
    if [ $ASYMMETRIC_DONE -eq 6 ]; then
        echo "✅ COMPLETE"
    else
        echo "⏳ RUNNING"
    fi

    echo "│"
    echo "└─────────────────────────────────────────────────────────────────────────┘"
    echo ""

    # Latest activity
    echo "┌─ LATEST ACTIVITY ───────────────────────────────────────────────────────┐"
    echo "│"

    # Pure latest
    if [ -f tournament_pure.log ]; then
        PURE_LATEST=$(grep "Running:" tournament_pure.log | tail -1 | sed 's/.*Running: //' | cut -c1-40)
        printf "│ Pure:       %-60s│\n" "$PURE_LATEST"
    fi

    # Pairwise latest
    if [ -f tournament_pairwise.log ]; then
        PAIR_LATEST=$(grep "Running:" tournament_pairwise.log | tail -1 | sed 's/.*Running: //' | cut -c1-40)
        printf "│ Pairwise:   %-60s│\n" "$PAIR_LATEST"
    fi

    # Asymmetric latest
    if [ -f tournament_asymmetric.log ]; then
        ASYM_LATEST=$(grep "Running:" tournament_asymmetric.log | tail -1 | sed 's/.*Running: //' | cut -c1-40)
        printf "│ Asymmetric: %-60s│\n" "$ASYM_LATEST"
    fi

    echo "│"
    echo "└─────────────────────────────────────────────────────────────────────────┘"
    echo ""

    # Check if all done
    if [ $TOTAL_DONE -eq $TOTAL ]; then
        echo "╔════════════════════════════════════════════════════════════════════════════╗"
        echo "║                     🎉 ALL TOURNAMENTS COMPLETE! 🎉                        ║"
        echo "╚════════════════════════════════════════════════════════════════════════════╝"
        echo ""
        echo "Run analysis:"
        echo "  python scripts/analyze_tournament_results.py --auto-discover"
        echo ""
        break
    fi

    echo "Press Ctrl+C to exit monitoring..."
    echo ""

    sleep 5
done
