| Rank | Model            | Family / Type         | MMLU %* | Input $/1M | Output $/1M | Notes |
|------|------------------|-----------------------|--------:|-----------:|------------:|-------|
| 1    | GPT-5            | GPT-5 frontier        | ~93.5   | 1.25       | 10.00       | Kaggle MMLU leaderboard; best public score for GPT-5. :contentReference[oaicite:3]{index=3} |
| 2    | o3               | o-series reasoning    | 93.4    | 2.00       | 8.00        | OpenAI open-model doc MMLU row. :contentReference[oaicite:4]{index=4} |
| 3    | o4-mini          | o-series reasoning    | 93.0    | 1.10       | 4.40        | Open-weight o4-mini; same architecture family as API o4-mini. :contentReference[oaicite:5]{index=5} |
| 4    | o1 (high)        | o-series reasoning    | 91.8    | 15.00      | 60.00       | From GPT-4.1 appendix table (same eval as 4.1/4o). :contentReference[oaicite:6]{index=6} |
| 5    | GPT-4.5 preview  | GPT-4.5 general       | 90.8    | 75.00      | 150.00      | MMLU in GPT-4.1 appendix; extremely expensive preview model. :contentReference[oaicite:7]{index=7} |
| 6    | GPT-4.1          | GPT-4.1 flagship      | 90.2    | 2.00       | 8.00        | New flagship general model; same table as above. :contentReference[oaicite:8]{index=8} |
| 7    | GPT-4.1 mini     | GPT-4.1 “small”       | 87.5    | 0.40       | 1.60        | Much cheaper than 4.1 with only small MMLU drop. :contentReference[oaicite:9]{index=9} |
| 8    | o3-mini (high)   | o-series reasoning    | 86.9    | 1.10       | 4.40        | Reasoning mini; matches / beats o1-class on many STEM tasks. :contentReference[oaicite:10]{index=10} |
| 9    | GPT-4            | GPT-4 (legacy)        | 86.4    | ~30        | ~60         | Original GPT-4 technical report MMLU ≈86.4%. :contentReference[oaicite:11]{index=11} |
| 10   | GPT-4o           | GPT-4o multimodal     | 85.7    | 2.50       | 10.00       | MMLU from 4.1 appendix (same harness); earlier “Hello 4o” reports 88.7 with different setup. :contentReference[oaicite:12]{index=12} |
| 11   | GPT-4o mini      | GPT-4o small          | 82.0    | 0.15       | 0.60        | MMLU 82% (Reuters + GPT-4.1 table). :contentReference[oaicite:13]{index=13} |
| 12   | GPT-4.1 nano     | GPT-4.1 ultra-small   | 80.1    | 0.10       | 0.40        | Cheapest general GPT-4-class model. :contentReference[oaicite:14]{index=14} |
| 13   | GPT-3.5-turbo    | GPT-3.5 chat          | ~70     | 0.50       | 1.50        | Widely reported ≈70% MMLU; very cheap legacy workhorse. :contentReference[oaicite:15]{index=15} |
| 14   | davinci-002      | GPT-3 “davinci”       | ~68     | 2.00       | 2.00        | Reddit reports MMLU ≈68 vs 70 for 3.5; treat as approximate. :contentReference[oaicite:16]{index=16} |
| 15   | babbage-002      | GPT-3 “babbage”       | n/a     | 0.40       | 0.40        | No solid public MMLU; much weaker than davinci/3.5 on most tasks. :contentReference[oaicite:17]{index=17} |


| Model           | Family / Type        | Published metric (not classic MMLU)          | Rough level vs ranked models | Input $/1M | Output $/1M | Note |
|----------------|----------------------|----------------------------------------------|------------------------------|-----------:|------------:|------|
| GPT-5.1        | GPT-5 “refresh”      | MMLU-Pro ≈86.4% (Vals)  | Roughly GPT-4.1 / o1-class on harder MMLU-Pro; classic MMLU not yet public. | 1.25 | 10.00 | Same pricing as GPT-5 in API. :contentReference[oaicite:20]{index=20} |
| GPT-5-mini     | GPT-5 small          | MMLU-Pro low-80s (Vals shows ≈82.5%). :contentReference[oaicite:21]{index=21} | Likely between GPT-4o and GPT-4.1-mini on classic MMLU; no official number yet. | 0.25 | 2.00 | Cheaper, long-context variant. :contentReference[oaicite:22]{index=22} |
| GPT-5-nano     | GPT-5 tiny           | Only partial MMLU-Pro / MMMU numbers public | Probably around GPT-4.1-nano / slightly below on MMLU; built for latency/cost. | 0.05 | 0.40 | API-only ultra-cheap tier. :contentReference[oaicite:23]{index=23} |
| o1-mini        | o-series small       | OpenAI reports o1-level performance on many reasoning tasks (AIME, GPQA, etc.), but not a clean MMLU row. :contentReference[oaicite:24]{index=24} | Comparable to GPT-4.1 on reasoning; we don’t have a single canonical MMLU %. | 1.10 | 4.40 | Same price as o3-mini/o4-mini. :contentReference[oaicite:25]{index=25} |
| o1-pro         | o-series high-end    | Stronger on hard math/coding; evals focus on AIME, Olympiads, etc., not MMLU. :contentReference[oaicite:26]{index=26} | Clearly above GPT-4.1 on hard reasoning, but no single MMLU figure. | 150.00 | 600.00 | Extremely expensive; niche “max reasoning” use. :contentReference[oaicite:27]{index=27} |
| o3-pro         | o3 “large”           | Internal / partner evals (SWE-Bench, GPQA, SimpleQA, etc.), not public MMLU. :contentReference[oaicite:28]{index=28} | Above o3 on reasoning benchmarks, but MMLU not disclosed. | 20.00 | 80.00 | Positioned as top-end reasoning API. :contentReference[oaicite:29]{index=29} |
