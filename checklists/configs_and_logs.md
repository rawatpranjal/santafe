# Configs and Logs Convention

---

## Config Naming Convention

### Pattern
`{part}_{set}_{strategy}_{env}.yaml`

### Parts
- `p1` = Part 1 (Foundational)
- `p2` = Part 2 (Santa Fe Tournament)
- `p3` = Part 3 (PPO)
- `p4` = Part 4 (LLM)

### Competitor Sets
- `ctrl` = Against Control (1 vs 7 ZIC)
- `self` = Against Self (Self-Play)
- `rr` = Round Robin Tournament (Mixed)

### Strategies
- `skel` = Skeleton
- `zic` = ZIC
- `zip` = ZIP
- `gd` = GD
- `kap` = Kaplan
- `ppo` = PPO
- `llm` = LLM

### Environments
- `base`, `bbbs`, `bsss`, `eql`, `ran`, `per`, `shrt`, `tok`, `sml`, `lad`

### Config Folder Structure
```
conf/experiment/
├── p1_foundational/
│   ├── p1_self_zi_{env}.yaml
│   ├── p1_self_zic_{env}.yaml
│   └── p1_self_zip_{env}.yaml
├── p2_tournament/
│   ├── ctrl/
│   │   └── p2_ctrl_{strategy}_{env}.yaml
│   ├── self/
│   │   └── p2_self_{strategy}_{env}.yaml
│   └── rr/
│       └── p2_rr_mixed_{env}.yaml
├── p3_ppo/
│   ├── train/
│   ├── ctrl/
│   ├── self/
│   └── rr/
└── p4_llm/
    ├── ctrl/
    ├── self/
    └── rr/
```

### Examples
- `p1_self_zic_base.yaml` = Part 1, ZIC self-play, BASE env
- `p2_ctrl_kap_base.yaml` = Part 2, Kaplan vs 7 ZIC, BASE env
- `p2_self_zip_ran.yaml` = Part 2, ZIP self-play, RAN env
- `p2_rr_mixed_eql.yaml` = Part 2, Round Robin mixed, EQL env
- `p3_self_ppo_shrt.yaml` = Part 3, PPO self-play, SHRT env
- `p4_ctrl_llm_base.yaml` = Part 4, LLM vs 7 ZIC, BASE env
- `p4_rr_llm_base.yaml` = Part 4, LLM in mixed, BASE env

---

## Experiment Logging Convention

### Log File Naming Pattern
`exp_{exp_id}_{strategy}_{env}_{timestamp}.json`

**Examples:**
- `exp_1.11_zic_base_20251128_143022.json`
- `exp_2.61_zip_base_20251128_150512.json`
- `exp_3.14_ppo_base_20251128_162345.json`

### Log Folder Structure
```
logs/
├── p1_foundational/
│   └── exp_1.xx_*.json
├── p2_tournament/
│   └── exp_2.xx_*.json
├── p3_ppo/
│   └── exp_3.xx_*.json
└── p4_llm/
    └── exp_4.xx_*.json
```

### Required Log Contents (for scientific replication)

Each log file must contain:

1. **Metadata**
   - `experiment_id`: e.g., "2.61"
   - `timestamp`: ISO 8601 format
   - `git_commit_hash`: full SHA
   - `random_seed`: int

2. **Configuration**
   - Full config dump (market params, agent params, hyperparameters)
   - Config file path used
   - Any overrides applied

3. **Environment**
   - Python version
   - Package versions (requirements freeze)
   - Hardware info (CPU, RAM, GPU if applicable)
   - OS version

4. **Results**
   - All metrics (efficiency, price RMSD, autocorrelation, etc.)
   - Per-period data
   - Trade-by-trade log (timestamp, buyer, seller, price, quantity)
   - Final rankings

5. **Checksums**
   - SHA256 of input config
   - SHA256 of output data

### Summary File
Each experiment also produces a human-readable summary:

`exp_{exp_id}_summary.md`

**Example:** `exp_2.61_summary.md`

Contents:
- Experiment ID and description
- Key configuration parameters
- Summary metrics table
- Notable observations

---

## Event Log (for Market Heartbeat Visualization)

Optional detailed event log for analyzing trader timing patterns.

### Enabling Event Logging

Add to config or command line:
```yaml
log_events: true
log_dir: logs/p1_foundational
experiment_id: exp_1.11
```

Or via command line:
```bash
python scripts/run_experiment.py experiment=p1_self_zic_base +log_events=true
```

### Event Log File

**Pattern:** `{experiment_id}_events.jsonl`

**Example:** `exp_1.11_events.jsonl`

### Event Log Format (JSONL)

One JSON object per line:

```jsonl
{"event_type":"bid_ask","round":1,"period":1,"step":1,"agent_id":1,"agent_type":"ZIC","is_buyer":true,"price":450,"status":"winner"}
{"event_type":"bid_ask","round":1,"period":1,"step":1,"agent_id":2,"agent_type":"Kaplan","is_buyer":true,"price":0,"status":"pass"}
{"event_type":"bid_ask","round":1,"period":1,"step":1,"agent_id":5,"agent_type":"ZIC","is_buyer":false,"price":520,"status":"winner"}
{"event_type":"trade","round":1,"period":1,"step":3,"buyer_id":1,"seller_id":5,"price":520}
```

### Event Types

**bid_ask**: A bid or ask submission
- `round`, `period`, `step`: Temporal coordinates
- `agent_id`: Global player ID
- `agent_type`: Strategy class name (e.g., "ZIC", "Kaplan")
- `is_buyer`: true for bid, false for ask
- `price`: Submitted price (0 if pass)
- `status`: "winner", "beaten", "pass", "standing", "tie_lost"

**trade**: A trade execution
- `round`, `period`, `step`: Temporal coordinates
- `buyer_id`, `seller_id`: Global player IDs
- `price`: Transaction price

### Visualization

```bash
# Plot Market Heartbeat
python scripts/visualize_heartbeat.py logs/exp_1.11_events.jsonl

# Save to file
python scripts/visualize_heartbeat.py logs/exp_1.11_events.jsonl -o figures/heartbeat.png

# Filter to specific round/period
python scripts/visualize_heartbeat.py logs/exp_1.11_events.jsonl --round 1 --period 1

# Print summary only
python scripts/visualize_heartbeat.py logs/exp_1.11_events.jsonl --summary
```
