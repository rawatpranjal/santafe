#!/usr/bin/env python3
import pandas as pd

df = pd.read_csv('results/validation_zip/results.csv')

print("="*60)
print("ZIP VALIDATION RESULTS AFTER FIX")
print("="*60)

print(f"\nOverall Average: {df['efficiency'].mean():.2f}%")
print(f"Expected: 85-95%")
print(f"Status: {'✅ PASS' if 85 <= df['efficiency'].mean() <= 95 else '❌ FAIL'}")

print(f"\nRounds with >0% efficiency: {(df.groupby('round')['efficiency'].mean() > 0).sum()}/50")
print(f"Rounds with 0% efficiency: {(df.groupby('round')['efficiency'].mean() == 0).sum()}/50")

print("\nFirst 20 Rounds:")
for r in range(1, 21):
    eff = df[df['round']==r]['efficiency'].mean()
    status = "✅" if eff > 80 else ("⚠️" if eff > 50 else "❌")
    print(f"Round {r:2d}: {eff:6.2f}% {status}")
