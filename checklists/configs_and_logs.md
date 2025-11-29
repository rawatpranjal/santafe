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
