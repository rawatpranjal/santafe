#!/bin/bash
# Tournament Monitoring Script
# Displays progress of all running tournament batches

echo "================================"
echo "Tournament Batch Status Monitor"
echo "================================"
echo ""

# Check Pure tournaments
echo "1. PURE SELF-PLAY (10 configs)"
if [ -f tournament_pure.log ]; then
    grep -E "Running:|SUCCESS|FAILED|Completed" tournament_pure.log | tail -5
    echo ""
else
    echo "  Not started"
    echo ""
fi

# Check Pairwise tournaments
echo "2. PAIRWISE (45 configs)"
if [ -f tournament_pairwise.log ]; then
    grep -E "Running:|SUCCESS|FAILED|Completed" tournament_pairwise.log | tail -5
    echo ""
else
    echo "  Not started"
    echo ""
fi

# Check Mixed tournaments
echo "3. MIXED STRATEGIES (8 configs)"
if [ -f tournament_mixed.log ]; then
    grep -E "Running:|SUCCESS|FAILED|Completed" tournament_mixed.log | tail -5
    echo ""
else
    echo "  Not started"
    echo ""
fi

# Check 1v7 tournaments
echo "4. INVASIBILITY/1v7 (20 configs)"
if [ -f tournament_1v7.log ]; then
    grep -E "Running:|SUCCESS|FAILED|Completed" tournament_1v7.log | tail -5
    echo ""
else
    echo "  Not started"
    echo ""
fi

# Summary
echo "================================"
echo "SUMMARY"
echo "================================"
echo "Pure:      $(grep -c SUCCESS tournament_pure.log 2>/dev/null || echo 0)/10 completed"
echo "Pairwise:  $(grep -c SUCCESS tournament_pairwise.log 2>/dev/null || echo 0)/45 completed"
echo "Mixed:     $(grep -c SUCCESS tournament_mixed.log 2>/dev/null || echo 0)/8 completed"
echo "1v7:       $(grep -c SUCCESS tournament_1v7.log 2>/dev/null || echo 0)/20 completed"
echo ""

# Check for failures
TOTAL_FAILURES=$(( $(grep -c FAILED tournament_pure.log 2>/dev/null || echo 0) + \
                   $(grep -c FAILED tournament_pairwise.log 2>/dev/null || echo 0) + \
                   $(grep -c FAILED tournament_mixed.log 2>/dev/null || echo 0) + \
                   $(grep -c FAILED tournament_1v7.log 2>/dev/null || echo 0) ))

if [ $TOTAL_FAILURES -gt 0 ]; then
    echo "⚠️  WARNING: $TOTAL_FAILURES failed experiments"
    echo ""
fi

echo "Total: $(($(grep -c SUCCESS tournament_*.log 2>/dev/null || echo 0)))/83 completed"
echo ""
echo "Last updated: $(date)"
